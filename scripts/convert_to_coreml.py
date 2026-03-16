#!/usr/bin/env python3
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
# Convert FreeTrace ONNX alpha models to CoreML .mlpackage format
# Run on macOS: python3 scripts/convert_to_coreml.py
#
# Prerequisites:
#   pip install coremltools==7.2 onnx
#   (coremltools 7.x supports direct ONNX conversion; 8+ dropped it)

import os
import sys

def convert_with_ct7(onnx_path, output_path, model_num):
    """Convert using coremltools 7.x direct ONNX support."""
    import coremltools as ct
    print(f"  Using coremltools {ct.__version__} ONNX converter...")

    model = ct.converters.onnx.convert(
        model=onnx_path,
        minimum_ios_deployment_target='15',
    )
    model.save(output_path)
    return True


def convert_with_ct_unified(onnx_path, output_path, model_num):
    """Convert using coremltools unified converter (needs source='auto' or torch)."""
    import coremltools as ct
    print(f"  Using coremltools {ct.__version__} unified converter...")

    model = ct.convert(
        onnx_path,
        compute_units=ct.ComputeUnit.ALL,
    )
    model.save(output_path)
    return True


def convert_via_torch(onnx_path, output_path, model_num):
    """Convert ONNX -> PyTorch -> CoreML."""
    import coremltools as ct
    import torch

    try:
        import onnx2torch
    except ImportError:
        print("  pip install onnx2torch")
        return False

    print(f"  Converting via PyTorch intermediary...")

    # Load ONNX as PyTorch model
    torch_model = onnx2torch.convert(onnx_path)
    torch_model.eval()

    # Trace with example input: [batch=1, seq_len=model_num, 1, 3]
    example = torch.randn(1, model_num, 1, 3)
    traced = torch.jit.trace(torch_model, example)

    # Convert to CoreML
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

        # Try methods in order of preference
        for method_name, method in [
            ("coremltools ONNX", convert_with_ct7),
            ("coremltools unified", convert_with_ct_unified),
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
        print(f"  pip install coremltools==7.2 onnx    # try older coremltools")
        print(f"  pip install onnx2torch               # for PyTorch intermediary")

    return 0 if success == len(model_nums) else 1


if __name__ == '__main__':
    sys.exit(main())
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
