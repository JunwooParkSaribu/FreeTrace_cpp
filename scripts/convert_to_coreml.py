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
import numpy as np  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16

# Monkey-patch for coremltools + NumPy 2.x compatibility  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
# Multiple functions removed in NumPy 2.0 but coremltools 7.x still uses them
if not hasattr(np, 'issubclass_'):
    np.issubclass_ = issubclass
if not hasattr(np, 'issctype'):
    np.issctype = lambda rep: isinstance(rep, type) and issubclass(rep, np.generic)
if not hasattr(np, 'obj2sctype'):
    np.obj2sctype = lambda rep, default=None: np.result_type(rep).type if rep is not None else default
if not hasattr(np, 'bool'):
    np.bool = np.bool_
if not hasattr(np, 'int'):
    np.int = np.int_
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'complex'):
    np.complex = np.complex128
if not hasattr(np, 'str'):
    np.str = np.str_  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16


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
    """Convert Keras model to CoreML. Tries multiple strategies."""
    import tensorflow as tf
    import coremltools as ct

    print(f"  Loading Keras model: {keras_path}")
    model = tf.keras.models.load_model(keras_path)
    model.summary()

    # Build input spec from model
    input_shape = model.input_shape  # e.g. (None, None, 1, 3)
    if input_shape[1] is None:
        spec = tf.TensorSpec(shape=[None, None, 1, 3], dtype=tf.float32, name='input')
    else:
        spec = tf.TensorSpec(shape=[None, input_shape[1], 1, 3], dtype=tf.float32, name='input')

    # Strategy 1: Convert concrete function directly (no SavedModel on disk)
    print(f"  Strategy 1: converting concrete function directly...")
    try:
        @tf.function(input_signature=[spec])
        def serve(x):
            return model(x, training=False)
        concrete_func = serve.get_concrete_function()
        mlmodel = ct.convert(
            [concrete_func],
            source='tensorflow',
            compute_units=ct.ComputeUnit.ALL,
        )
        mlmodel.save(output_path)
        print(f"  -> OK (strategy 1): {output_path}")
        return True
    except Exception as e:
        print(f"  Strategy 1 failed: {e}")

    # Strategy 2: Pass Keras model object directly to coremltools
    print(f"  Strategy 2: converting Keras model object directly...")
    try:
        mlmodel = ct.convert(
            model,
            source='tensorflow',
            compute_units=ct.ComputeUnit.ALL,
        )
        mlmodel.save(output_path)
        print(f"  -> OK (strategy 2): {output_path}")
        return True
    except Exception as e:
        print(f"  Strategy 2 failed: {e}")

    # Strategy 3: Use model.export() then convert with ct.convert on the dir
    # (the "multiple concrete functions" error — try with minimum_deployment_target)
    import shutil
    saved_model_dir = keras_path.replace('.keras', '_saved_model')
    print(f"  Strategy 3: export SavedModel + minimum_deployment_target...")
    try:
        model.export(saved_model_dir)
        mlmodel = ct.convert(
            saved_model_dir,
            source='tensorflow',
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS13,
        )
        mlmodel.save(output_path)
        shutil.rmtree(saved_model_dir, ignore_errors=True)
        print(f"  -> OK (strategy 3): {output_path}")
        return True
    except Exception as e:
        print(f"  Strategy 3 failed: {e}")
        shutil.rmtree(saved_model_dir, ignore_errors=True)

    return False # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16


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
