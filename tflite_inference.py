"""
TensorFlow Lite inference helpers for lightweight model loading.
"""

import numpy as np
from pathlib import Path
import logging

try:
    # Preferred lightweight runtime for deployment.
    from tflite_runtime.interpreter import Interpreter
except Exception:  # pragma: no cover - fallback for local environments
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

logger = logging.getLogger(__name__)


class TFLiteInference:
    """Wrapper for TFLite model inference"""
    
    def __init__(self, model_path: str):
        """Load TFLite model"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output tensor details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        logger.info(f"✅ TFLite model loaded: {model_path}")
        logger.info(f"   Input: {self.input_details[0]['shape']}")
        logger.info(f"   Output: {self.output_details[0]['shape']}")
    
    def predict(self, image_array: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed image.
        
        Args:
            image_array: Preprocessed image (224, 224, 3) normalized to [0, 1]
        
        Returns:
            Model output (predictions)
        """
        # Ensure correct shape and dtype
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        image_array = image_array.astype(np.float32)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], image_array)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return output
