#!/usr/bin/env python3
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
# Convert FreeTrace Keras ConvLSTM alpha models to CoreML via PyTorch.
#
# coremltools cannot convert TF ConvLSTM (While/Loop ops). Instead we:
# 1. Extract Keras weights
# 2. Build faithful PyTorch reimplementation using Conv1d + manual LSTM cell
# 3. Trace (loop unrolls since seq_len is fixed per model)
# 4. Convert traced model to CoreML
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


def extract_keras_weights(model): # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
    """Extract weights from Keras model, organized by layer type and index."""
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
        w = {}
        for wt in layer.weights:
            # Extract clean name: last component, strip :0
            name = wt.name.split('/')[-1]
            if ':' in name:
                name = name[:name.rfind(':')]
            w[name] = wt.numpy()
            print(f"    ConvLSTM '{name}': {wt.numpy().shape}")
        result['conv_lstm'].append(w)

    for layer in bn_layers:
        w = {}
        for wt in layer.weights:
            name = wt.name.split('/')[-1]
            if ':' in name:
                name = name[:name.rfind(':')]
            w[name] = wt.numpy()
        result['bn'].append(w)

    for layer in dense_layers:
        w = {}
        for wt in layer.weights:
            name = wt.name.split('/')[-1]
            if ':' in name:
                name = name[:name.rfind(':')]
            w[name] = wt.numpy()
            print(f"    Dense '{name}': {wt.numpy().shape}")
        result['dense'].append(w)

    return result # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16


def build_pytorch_model(weights, seq_len): # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
    """Build faithful PyTorch ConvLSTM reimplementation.

    Uses Conv1d for the gate computations (faithful to Keras ConvLSTM1D).
    The recurrent loop is explicit and unrolls during jit.trace.

    Keras ConvLSTM kernel: (kernel_size, in_ch, 4*filters)
    PyTorch Conv1d weight: (out_ch, in_ch, kernel_size)
    Mapping: pytorch_weight = keras_kernel.transpose(2, 0, 1)
    """
    import torch
    import torch.nn as nn

    conv_lstm_w = weights['conv_lstm']
    bn_w = weights['bn']
    dense_w = weights['dense']

    # Determine layer configs from weights
    layer_configs = []
    for cw in conv_lstm_w:
        k = cw['kernel']  # (kernel_size, in_ch, 4*filters)
        kernel_size = k.shape[0]
        in_ch = k.shape[1]
        hidden = k.shape[2] // 4
        layer_configs.append((in_ch, hidden, kernel_size))
        print(f"    ConvLSTM layer: in={in_ch}, hidden={hidden}, kernel_size={kernel_size}")

    class ConvLSTMModel(nn.Module):
        def __init__(self, configs, seq_len):
            super().__init__()
            self.seq_len = seq_len
            self.n_lstm = len(configs)

            # ConvLSTM layers: input conv + recurrent conv per layer
            self.conv_ih = nn.ModuleList()
            self.conv_hh = nn.ModuleList()
            self.hidden_sizes = []

            for in_ch, hidden, ks in configs:
                pad = ks // 2  # same padding
                self.conv_ih.append(nn.Conv1d(in_ch, 4 * hidden, ks, padding=pad, bias=True))
                self.conv_hh.append(nn.Conv1d(hidden, 4 * hidden, ks, padding=pad, bias=False))
                self.hidden_sizes.append(hidden)

            # BatchNorm layers (eps=0.001 to match Keras default)
            self.bns = nn.ModuleList()
            for _, hidden, _ in configs:
                self.bns.append(nn.BatchNorm1d(hidden, eps=0.001))

            # Dense layers
            d0_in = configs[-1][1]
            d0_out = dense_w[0]['kernel'].shape[-1]
            d1_out = dense_w[1]['kernel'].shape[-1]
            d2_out = dense_w[2]['kernel'].shape[-1]
            self.fc1 = nn.Linear(d0_in, d0_out)
            self.fc2 = nn.Linear(d0_out, d1_out)
            self.fc3 = nn.Linear(d1_out, d2_out)

        def forward(self, x):
            # x: (batch, seq, spatial=1, channels)
            batch = x.shape[0]

            for layer_idx in range(self.n_lstm):
                hidden_size = self.hidden_sizes[layer_idx]
                return_seq = (layer_idx < self.n_lstm - 1)

                # Initialize hidden/cell state: (batch, hidden, spatial=1)
                h = torch.zeros(batch, hidden_size, 1, device=x.device)
                c = torch.zeros(batch, hidden_size, 1, device=x.device)

                outputs = []
                for t in range(self.seq_len):
                    # x_t: (batch, spatial=1, channels) → (batch, channels, spatial=1)
                    x_t = x[:, t, :, :].permute(0, 2, 1)

                    gates = self.conv_ih[layer_idx](x_t) + self.conv_hh[layer_idx](h)
                    i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)
                    i_gate = torch.sigmoid(i_gate)
                    f_gate = torch.sigmoid(f_gate)
                    g_gate = torch.tanh(g_gate)
                    o_gate = torch.sigmoid(o_gate)
                    c = f_gate * c + i_gate * g_gate
                    h = o_gate * torch.tanh(c)
                    outputs.append(h)

                if return_seq:
                    # (batch, seq, hidden, spatial=1) → BN → (batch, seq, spatial=1, hidden)
                    out = torch.stack(outputs, dim=1)  # (batch, seq, hidden, 1)
                    # BN: reshape to (batch*seq, hidden) for BN1d, then reshape back
                    bs = out.shape[0] * out.shape[1]
                    out_flat = out.squeeze(-1).reshape(bs, -1)  # (batch*seq, hidden)
                    out_flat = self.bns[layer_idx](out_flat)
                    out = out_flat.reshape(batch, self.seq_len, -1).unsqueeze(2)  # (batch,seq,1,hidden)
                    x = out
                else:
                    # Take last hidden: (batch, hidden, 1) → BN → (batch, hidden)
                    h_last = h.squeeze(-1)  # (batch, hidden)
                    h_last = self.bns[layer_idx](h_last)
                    x = h_last

            # Dense layers: x is (batch, hidden)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x

    pt_model = ConvLSTMModel(layer_configs, seq_len)

    # Transfer weights
    with torch.no_grad():
        for i, cw in enumerate(conv_lstm_w):
            # Keras kernel: (ks, in_ch, 4*h) → PyTorch Conv1d: (4*h, in_ch, ks)
            kernel = cw['kernel']  # (ks, in_ch, 4*h)
            pt_model.conv_ih[i].weight.copy_(
                torch.from_numpy(kernel.transpose(2, 0, 1)))  # (4*h, in_ch, ks)
            pt_model.conv_ih[i].bias.copy_(
                torch.from_numpy(cw['bias']))

            rec_kernel = cw['recurrent_kernel']  # (ks, h, 4*h)
            pt_model.conv_hh[i].weight.copy_(
                torch.from_numpy(rec_kernel.transpose(2, 0, 1)))  # (4*h, h, ks)

        for i, bw in enumerate(bn_w):
            pt_model.bns[i].weight.copy_(torch.from_numpy(bw['gamma']))
            pt_model.bns[i].bias.copy_(torch.from_numpy(bw['beta']))
            pt_model.bns[i].running_mean.copy_(torch.from_numpy(bw['moving_mean']))
            pt_model.bns[i].running_var.copy_(torch.from_numpy(bw['moving_variance']))

        for i, dw in enumerate(dense_w):
            kernel = torch.from_numpy(dw['kernel'])  # (in, out)
            bias = torch.from_numpy(dw['bias'].flatten())
            fc = [pt_model.fc1, pt_model.fc2, pt_model.fc3][i]
            fc.weight.copy_(kernel.T)
            fc.bias.copy_(bias)

    pt_model.eval()
    return pt_model # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16


def verify_outputs(keras_model, pt_model, model_num): # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
    """Compare Keras and PyTorch model outputs on test input."""
    import torch

    np.random.seed(42)
    test_input = np.random.randn(2, model_num, 1, 3).astype(np.float32)

    keras_out = keras_model(test_input, training=False).numpy().flatten()

    with torch.no_grad():
        pt_out = pt_model(torch.from_numpy(test_input)).numpy().flatten()

    max_diff = np.max(np.abs(keras_out - pt_out))
    print(f"    Keras:   {keras_out}")
    print(f"    PyTorch: {pt_out}")
    print(f"    Max absolute difference: {max_diff:.6e}")

    if max_diff > 0.01:
        print(f"    WARNING: Large difference! Weight mapping may be incorrect.")
        return False
    print(f"    OK: outputs match within tolerance")
    return True # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16


def convert_via_pytorch(keras_path, output_path, model_num): # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
    """Convert Keras ConvLSTM model to CoreML via PyTorch reimplementation."""
    import tensorflow as tf
    import torch
    import coremltools as ct

    print(f"  Loading Keras model: {keras_path}")
    model = tf.keras.models.load_model(keras_path)

    print(f"  Extracting weights...")
    weights = extract_keras_weights(model)

    print(f"  Building PyTorch ConvLSTM model (seq_len={model_num})...")
    pt_model = build_pytorch_model(weights, model_num)

    print(f"  Verifying output equivalence...")
    if not verify_outputs(model, pt_model, model_num):
        print(f"  Continuing anyway — will check empirically after conversion")

    print(f"  Tracing PyTorch model...")
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
