#!/usr/bin/env python3
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
# Convert FreeTrace Keras ConvLSTM alpha models to CoreML via PyTorch intermediate.
#
# The coremltools TF converter cannot handle ConvLSTM (While/Loop ops).
# But ConvLSTM1D with kernel_size=1, spatial_dim=1 is mathematically identical
# to a standard LSTM. So we: Keras weights → PyTorch nn.LSTM → CoreML.
#
# Prerequisites:
#   pip install coremltools tensorflow torch
#   FreeTrace Python package installed (pip install FreeTrace)

import os
import sys
import glob
import numpy as np

# NumPy 2.x compat patches for coremltools 7.x
if not hasattr(np, 'issubclass_'):
    np.issubclass_ = issubclass
if not hasattr(np, 'issctype'):
    np.issctype = lambda rep: isinstance(rep, type) and issubclass(rep, np.generic)
if not hasattr(np, 'obj2sctype'):
    np.obj2sctype = lambda rep, default=None: np.result_type(rep).type if rep is not None else default


def find_keras_models():
    """Find .keras model files from installed FreeTrace package or local."""
    try:
        import FreeTrace
        pkg_dir = os.path.dirname(FreeTrace.__file__)
        models_dir = os.path.join(pkg_dir, 'models')
        if os.path.isdir(models_dir):
            return models_dir
    except ImportError:
        pass

    for pattern in [
        os.path.expanduser('~/claude/claude_venv/lib/python*/site-packages/FreeTrace/models'),
        os.path.expanduser('~/venv/lib/python*/site-packages/FreeTrace/models'),
        os.path.expanduser('~/*/FreeTrace/models'),
    ]:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


def extract_keras_weights(model):
    """Extract weights from Keras model, organized by layer type and index."""
    import tensorflow as tf

    conv_lstm_layers = []
    bn_layers = []
    dense_layers = []

    for layer in model.layers:
        ltype = type(layer).__name__
        if 'ConvLSTM' in ltype:
            conv_lstm_layers.append(layer)
        elif 'BatchNormalization' in ltype:
            bn_layers.append(layer)
        elif 'Dense' in ltype:
            dense_layers.append(layer)

    print(f"    Found: {len(conv_lstm_layers)} ConvLSTM, {len(bn_layers)} BN, {len(dense_layers)} Dense")

    result = {'conv_lstm': [], 'bn': [], 'dense': []}

    for layer in conv_lstm_layers:
        w = {wt.name.split('/')[-1].rstrip(':0'): wt.numpy() for wt in layer.weights}
        # Print shapes for debugging
        for name, arr in w.items():
            print(f"    ConvLSTM weight '{name}': {arr.shape}")
        result['conv_lstm'].append(w)

    for layer in bn_layers:
        w = {wt.name.split('/')[-1].rstrip(':0'): wt.numpy() for wt in layer.weights}
        result['bn'].append(w)

    for layer in dense_layers:
        w = {wt.name.split('/')[-1].rstrip(':0'): wt.numpy() for wt in layer.weights}
        result['dense'].append(w)

    return result


def build_pytorch_model(weights):
    """Build equivalent PyTorch model and transfer Keras weights.

    ConvLSTM1D(kernel_size=1, spatial_dim=1) == nn.LSTM mathematically.

    Keras ConvLSTM kernel:           (1, 1, input_ch, 4*filters) → squeeze → (in, 4*h)
    Keras ConvLSTM recurrent_kernel: (1, 1, filters,  4*filters) → squeeze → (h, 4*h)
    PyTorch LSTM weight_ih: (4*hidden, input)  — transposed from Keras
    PyTorch LSTM weight_hh: (4*hidden, hidden) — transposed from Keras

    Gate order: both use [i, f, g/c, o].
    """
    import torch
    import torch.nn as nn

    conv_lstm_w = weights['conv_lstm']
    bn_w = weights['bn']
    dense_w = weights['dense']

    # Determine layer sizes from weights  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
    # Kernel shape: (kernel_size, input_ch, 4*filters)
    # With spatial_dim=1 and padding='same', only the center element of the
    # kernel contributes (rest multiply zero-padded positions).
    lstm_configs = []
    for cw in conv_lstm_w:
        kernel = cw['kernel']  # (kernel_size, in_ch, 4*h)
        in_size = kernel.shape[1]      # input channels
        hidden_4 = kernel.shape[2]     # 4 * filters
        hidden_size = hidden_4 // 4
        lstm_configs.append((in_size, hidden_size))
        print(f"    LSTM layer: input={in_size}, hidden={hidden_size} (kernel_size={kernel.shape[0]})")  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16

    # Build model
    class AlphaModel(nn.Module):
        def __init__(self, configs):
            super().__init__()
            self.lstms = nn.ModuleList()
            self.bns = nn.ModuleList()
            for i, (inp, hid) in enumerate(configs):
                self.lstms.append(nn.LSTM(inp, hid, batch_first=True))
                self.bns.append(nn.BatchNorm1d(hid))

            # Dense layers — kernel shape is (in, out), don't squeeze
            d0_in = configs[-1][1]  # last LSTM hidden size
            d0_out = dense_w[0]['kernel'].shape[-1]
            d1_out = dense_w[1]['kernel'].shape[-1]
            d2_out = dense_w[2]['kernel'].shape[-1]
            self.fc1 = nn.Linear(d0_in, d0_out)
            self.fc2 = nn.Linear(d0_out, d1_out)
            self.fc3 = nn.Linear(d1_out, d2_out)

        def forward(self, x):
            # x: (batch, seq, 1, 3) → (batch, seq, 3)
            x = x.squeeze(2)

            # LSTM + BN layers (all return_sequences=True except last)
            for i, (lstm, bn) in enumerate(zip(self.lstms, self.bns)):
                x, _ = lstm(x)
                if i < len(self.lstms) - 1:
                    # return_sequences=True: BN over (batch, seq, hidden)
                    x = bn(x.transpose(1, 2)).transpose(1, 2)
                else:
                    # return_sequences=False: take last timestep
                    x = x[:, -1, :]  # (batch, hidden)
                    x = bn(x)

            # Dense layers
            x = self.fc1(x)
            x = self.fc2(x)
            # dropout is identity at inference
            x = self.fc3(x)
            return x

    pt_model = AlphaModel(lstm_configs)

    # Transfer weights
    with torch.no_grad():
        for i, cw in enumerate(conv_lstm_w):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
            # kernel: (kernel_size, in_ch, 4*h) — take center slice
            k = cw['kernel']
            center = k.shape[0] // 2
            kernel = torch.from_numpy(k[center])                         # (in, 4*h)
            rk = cw['recurrent_kernel']
            rc = rk.shape[0] // 2
            rec_kernel = torch.from_numpy(rk[rc])                        # (h, 4*h)
            bias = torch.from_numpy(cw['bias'])                          # (4*h,)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16

            pt_model.lstms[i].weight_ih_l0.copy_(kernel.T)      # (4*h, in)
            pt_model.lstms[i].weight_hh_l0.copy_(rec_kernel.T)  # (4*h, h)
            pt_model.lstms[i].bias_ih_l0.copy_(bias)
            pt_model.lstms[i].bias_hh_l0.zero_()  # Keras has single bias

        for i, bw in enumerate(bn_w):
            pt_model.bns[i].weight.copy_(torch.from_numpy(bw['gamma']))
            pt_model.bns[i].bias.copy_(torch.from_numpy(bw['beta']))
            pt_model.bns[i].running_mean.copy_(torch.from_numpy(bw['moving_mean']))
            pt_model.bns[i].running_var.copy_(torch.from_numpy(bw['moving_variance']))

        for i, dw in enumerate(dense_w):
            kernel = torch.from_numpy(dw['kernel'])  # (in, out)
            bias = torch.from_numpy(dw['bias'].flatten())
            fc = [pt_model.fc1, pt_model.fc2, pt_model.fc3][i]
            fc.weight.copy_(kernel.T)  # PyTorch: (out, in)
            fc.bias.copy_(bias)

    pt_model.eval()
    return pt_model


def verify_outputs(keras_model, pt_model, model_num):
    """Compare Keras and PyTorch model outputs on test input."""
    import tensorflow as tf
    import torch

    np.random.seed(42)
    test_input = np.random.randn(2, model_num, 1, 3).astype(np.float32)

    # Keras prediction
    keras_out = keras_model(test_input, training=False).numpy().flatten()

    # PyTorch prediction
    with torch.no_grad():
        pt_out = pt_model(torch.from_numpy(test_input)).numpy().flatten()

    max_diff = np.max(np.abs(keras_out - pt_out))
    print(f"    Verification: Keras={keras_out}, PyTorch={pt_out}")
    print(f"    Max absolute difference: {max_diff:.6e}")

    if max_diff > 0.01:
        print(f"    WARNING: Large difference! Weight mapping may be incorrect.")
        return False
    print(f"    OK: outputs match within tolerance")
    return True


def convert_via_pytorch(keras_path, output_path, model_num): # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
    """Convert Keras ConvLSTM model to CoreML via PyTorch LSTM intermediate."""
    import tensorflow as tf
    import torch
    import coremltools as ct

    print(f"  Loading Keras model: {keras_path}")
    model = tf.keras.models.load_model(keras_path)

    print(f"  Extracting weights...")
    weights = extract_keras_weights(model)

    print(f"  Building equivalent PyTorch model...")
    pt_model = build_pytorch_model(weights)

    print(f"  Verifying output equivalence...")
    if not verify_outputs(model, pt_model, model_num):
        print(f"  Verification failed — aborting conversion")
        return False

    print(f"  Tracing PyTorch model (seq_len={model_num})...")
    example_input = torch.randn(1, model_num, 1, 3)
    traced = torch.jit.trace(pt_model, example_input)

    print(f"  Converting to CoreML (compute_units=ALL for GPU/ANE)...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name='input',
            shape=ct.Shape(shape=(1, model_num, 1, 3)),
        )],
        compute_units=ct.ComputeUnit.ALL,
    )
    mlmodel.save(output_path)
    print(f"  -> OK: {output_path}")
    return True # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16


def main():
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    keras_dir = find_keras_models()
    if not keras_dir:
        print("ERROR: Cannot find FreeTrace Keras models.")
        print("Install FreeTrace: pip install FreeTrace")
        print("Or specify path: KERAS_MODELS_DIR=/path/to/models python3 scripts/convert_to_coreml.py")
        return 1

    keras_dir = os.environ.get('KERAS_MODELS_DIR', keras_dir)
    print(f"Keras models directory: {keras_dir}")
    print(f"Output directory: {output_dir}")

    model_nums = [3, 5, 8]
    success = 0

    for n in model_nums:
        keras_path = os.path.join(keras_dir, f'reg_model_{n}.keras')
        output_path = os.path.join(output_dir, f'reg_model_{n}.mlpackage')

        if not os.path.exists(keras_path):
            print(f"\nSkipping reg_model_{n}: {keras_path} not found")
            continue

        print(f"\nConverting reg_model_{n}...")
        try:
            if convert_via_pytorch(keras_path, output_path, n):
                success += 1
        except Exception as e:
            import traceback
            print(f"  -> FAILED: {e}")
            traceback.print_exc()

    print(f"\n{'='*40}")
    print(f"Converted {success}/{len(model_nums)} models")
    if success > 0:
        print(f"CoreML models saved to: {output_dir}/")
        print(f"Rebuild FreeTrace C++ to use GPU/ANE acceleration.")

    return 0 if success == len(model_nums) else 1


if __name__ == '__main__':
    sys.exit(main())
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
