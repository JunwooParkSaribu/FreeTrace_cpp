#!/usr/bin/env python3
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
# Convert FreeTrace ONNX alpha models to CoreML .mlpackage format
# Run on macOS: python3 scripts/convert_to_coreml.py
#
# Prerequisites:
#   pip install coremltools onnx onnx-tf tensorflow

import os
import sys
import shutil


def convert_via_tensorflow(onnx_path, output_path, model_num):
    """Convert ONNX -> TensorFlow SavedModel -> CoreML.
    Most reliable for ConvLSTM models (Loop ops convert back to tf.while_loop)."""
    import onnx
    print(f"  Converting ONNX -> TF SavedModel -> CoreML...")

    # Step 1: ONNX -> TF SavedModel
    from onnx_tf.backend import prepare
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)

    tf_dir = onnx_path.replace('.onnx', '_tf_saved')
    tf_rep.export_graph(tf_dir)
    print(f"  -> TF SavedModel: {tf_dir}")

    # Step 2: TF SavedModel -> CoreML
    import coremltools as ct
    mlmodel = ct.convert(
        tf_dir,
        source='tensorflow',
        compute_units=ct.ComputeUnit.ALL,
    )
    mlmodel.save(output_path)

    # Cleanup temp SavedModel
    shutil.rmtree(tf_dir, ignore_errors=True)
    return True


def convert_via_torch(onnx_path, output_path, model_num):
    """Convert ONNX -> PyTorch -> CoreML (fallback)."""
    import coremltools as ct
    import torch
    import onnx2torch

    print(f"  Converting ONNX -> PyTorch -> CoreML...")

    torch_model = onnx2torch.convert(onnx_path)
    torch_model.eval()

    example = torch.randn(1, model_num, 1, 3)
    traced = torch.jit.trace(torch_model, example)

    model = ct.convert(
        traced,
        source='pytorch',
        inputs=[ct.TensorType(name='input', shape=(ct.RangeDim(1, 1024), model_num, 1, 3))],
        compute_units=ct.ComputeUnit.ALL,
    )
    model.save(output_path)
    return True


def main():
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    models_dir = os.path.abspath(models_dir)

    model_nums = [3, 5, 8]
    success = 0

    for n in model_nums:
        onnx_path = os.path.join(models_dir, f'reg_model_{n}.onnx')
        output_path = os.path.join(models_dir, f'reg_model_{n}.mlpackage')

        if not os.path.exists(onnx_path):
            print(f"Skipping reg_model_{n}: {onnx_path} not found")
            continue

        print(f"\nConverting reg_model_{n}...")

        for method_name, method in [
            ("TensorFlow intermediary", convert_via_tensorflow),
            ("PyTorch intermediary", convert_via_torch),
        ]:
            try:
                if method(onnx_path, output_path, n):
                    print(f"  -> OK: {output_path}")
                    success += 1
                    break
            except Exception as e:
                print(f"  -> {method_name} failed: {e}")
                continue
        else:
            print(f"  -> ALL METHODS FAILED for reg_model_{n}")

    print(f"\n{'='*40}")
    print(f"Converted {success}/{len(model_nums)} models")
    if success > 0:
        print(f"CoreML models saved to: {models_dir}/")
        print(f"Rebuild FreeTrace to use them automatically.")
    else:
        print(f"\nTroubleshooting:")
        print(f"  pip install coremltools onnx onnx-tf tensorflow")

    return 0 if success == len(model_nums) else 1


if __name__ == '__main__':
    sys.exit(main())
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
