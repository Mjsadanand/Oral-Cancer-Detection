"""
Convert .h5 model to TensorFlow Lite format for deployment.
Run this once locally to generate the .tflite model file.
"""

import tensorflow as tf
from pathlib import Path
import sys
import tempfile

MODEL_PATH = "models/oral_cancer_model.h5"
TFLITE_PATH = "models/oral_cancer_model.tflite"

def convert_to_tflite():
    """Convert Keras .h5 model to TensorFlow Lite format"""
    
    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        sys.exit(1)
    
    print(f"📦 Loading model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"✅ Model loaded successfully!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    print(f"\n🔄 Converting to TensorFlow Lite format...")

    tflite_model = None
    conversion_errors = []

    # Attempt 1: basic Keras conversion (most compatible)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()
        print("✅ Conversion successful using Keras direct conversion")
    except Exception as e:
        conversion_errors.append(f"Keras direct conversion failed: {e}")

    # Attempt 2: SavedModel conversion fallback
    if tflite_model is None:
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                saved_model_dir = Path(tmp_dir) / "saved_model"
                tf.saved_model.save(model, str(saved_model_dir))

                converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS,
                ]
                converter._experimental_lower_tensor_list_ops = False
                tflite_model = converter.convert()
                print("✅ Conversion successful using SavedModel fallback")
        except Exception as e:
            conversion_errors.append(f"SavedModel fallback failed: {e}")

    # Attempt 2b: Keras export() + SavedModel conversion (Keras 3 friendly)
    if tflite_model is None:
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                exported_dir = Path(tmp_dir) / "exported_saved_model"
                model.export(str(exported_dir))

                converter = tf.lite.TFLiteConverter.from_saved_model(str(exported_dir))
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS,
                ]
                converter._experimental_lower_tensor_list_ops = False
                tflite_model = converter.convert()
                print("✅ Conversion successful using model.export fallback")
        except Exception as e:
            conversion_errors.append(f"model.export fallback failed: {e}")

    # Attempt 3: float16 optimization (sometimes fixes op issues)
    if tflite_model is None:
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter._experimental_lower_tensor_list_ops = False
            tflite_model = converter.convert()
            print("✅ Conversion successful using float16 fallback")
        except Exception as e:
            conversion_errors.append(f"Float16 fallback failed: {e}")

    # Attempt 4: concrete function conversion
    if tflite_model is None:
        try:
            input_shape = [1] + list(model.input_shape[1:])
            concrete_fn = tf.function(model).get_concrete_function(
                tf.TensorSpec(input_shape, tf.float32)
            )

            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], model)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter._experimental_lower_tensor_list_ops = False
            tflite_model = converter.convert()
            print("✅ Conversion successful using concrete function fallback")
        except Exception as e:
            conversion_errors.append(f"Concrete function fallback failed: {e}")

    if tflite_model is None:
        print("❌ Conversion failed after all fallback strategies.")
        for err in conversion_errors:
            print(f"   - {err}")
        sys.exit(1)
    
    # Save TFLite model
    tflite_path = Path(TFLITE_PATH)
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    
    # Compare sizes
    h5_size_mb = model_file.stat().st_size / (1024 * 1024)
    tflite_size_mb = tflite_path.stat().st_size / (1024 * 1024)
    
    print(f"\n📊 File sizes:")
    print(f"   Original (.h5):      {h5_size_mb:.2f} MB")
    print(f"   Converted (.tflite): {tflite_size_mb:.2f} MB")
    print(f"   Reduction:           {(1 - tflite_size_mb/h5_size_mb) * 100:.1f}%")
    
    print(f"\n✅ TFLite model saved to: {TFLITE_PATH}")
    print(f"\nNext step: Use this model in app.py for lightweight inference!")

if __name__ == "__main__":
    convert_to_tflite()
