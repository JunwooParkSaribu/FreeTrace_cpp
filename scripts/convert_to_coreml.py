#!/usr/bin/env python3
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
# Convert FreeTrace Keras alpha models to CoreML .mlpackage format
# Run on macOS: python3 scripts/convert_to_coreml.py
#
# Prerequisites:
#   pip install coremltools tensorflow
#   FreeTrace Python package installed (pip install FreeTrace)

import os
import sys
import glob


def find_keras_models():
    """Find .keras model files from installed FreeTrace package or local."""
    # Check installed FreeTrace package
    try:
        import FreeTrace
        pkg_dir = os.path.dirname(FreeTrace.__file__)
        models_dir = os.path.join(pkg_dir, 'models')
        if os.path.isdir(models_dir):
            return models_dir
    except ImportError:
        pass

    # Check common locations
    for pattern in [
        os.path.expanduser('~/claude/claude_venv/lib/python*/site-packages/FreeTrace/models'),
        os.path.expanduser('~/*/FreeTrace/models'),
    ]:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

    return None


def convert_keras_to_coreml(keras_path, output_path, model_num): # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
    """Convert Keras model to CoreML via TF SavedModel intermediary."""
    import tensorflow as tf
    import coremltools as ct
    import tempfile, shutil

    print(f"  Loading Keras model: {keras_path}")
    model = tf.keras.models.load_model(keras_path)
    model.summary()

    # Save as TF SavedModel first (avoids TF 2.18 / coremltools incompatibility)
    saved_model_dir = keras_path.replace('.keras', '_saved_model')
    print(f"  Exporting to TF SavedModel: {saved_model_dir}")
    model.export(saved_model_dir)

    print(f"  Converting SavedModel to CoreML (compute_units=ALL for GPU/ANE)...")
    mlmodel = ct.convert(
        saved_model_dir,
        source='tensorflow',
        compute_units=ct.ComputeUnit.ALL,
    )
    mlmodel.save(output_path)

    # Cleanup temp SavedModel
    shutil.rmtree(saved_model_dir, ignore_errors=True)
    print(f"  -> OK: {output_path}")
    return True # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16


def main():
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    output_dir = os.path.abspath(output_dir)

    # Find Keras source models
    keras_dir = find_keras_models()
    if not keras_dir:
        print("ERROR: Cannot find FreeTrace Keras models.")
        print("Install FreeTrace: pip install FreeTrace")
        print("Or specify path: KERAS_MODELS_DIR=/path/to/models python3 scripts/convert_to_coreml.py")
        return 1

    # Allow override via env var
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
            if convert_keras_to_coreml(keras_path, output_path, n):
                success += 1
        except Exception as e:
            print(f"  -> FAILED: {e}")

    print(f"\n{'='*40}")
    print(f"Converted {success}/{len(model_nums)} models")
    if success > 0:
        print(f"CoreML models saved to: {output_dir}/")
        print(f"Rebuild FreeTrace C++ to use GPU/ANE acceleration.")

    return 0 if success == len(model_nums) else 1


if __name__ == '__main__':
    sys.exit(main())
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
