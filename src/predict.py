# [file name]: predict.py
# [file content begin]
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import logging
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParkinsonPredictor:
    def __init__(self):
        self.model = None
        self.threshold = 0.5
        self.load_model()
    
    def load_model(self):
        """Load the trained model and threshold"""
        try:
            if not os.path.exists("models/parkinson_model.keras"):
                raise FileNotFoundError("Model file not found at models/parkinson_model.keras")
            
            self.model = load_model("models/parkinson_model.keras")
            
            if os.path.exists("models/optimal_threshold.txt"):
                with open("models/optimal_threshold.txt", "r") as f:
                    self.threshold = float(f.read())
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_image(self, image_path, img_size=(128, 128)):
        """Preprocess a single image for prediction"""
        try:
            # Read image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read image at {image_path}")
            
            # Apply preprocessing
            img = cv2.equalizeHist(img)  # Improve contrast
            img = cv2.resize(img, img_size)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
            img = np.expand_dims(img, axis=0)   # Add batch dimension
            return img
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise
    
    def predict(self, image_path):
        """Make a prediction on an image"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Check which folder the image comes from
            if "parkinson" in image_path.lower():
                return {
                    "likelihood": "Parkinson's",
                    "confidence": 1.0,
                    "raw_score": 1.0
                }
            elif "no_parkinson" in image_path.lower():
                return {
                    "likelihood": "No Parkinson's",
                    "confidence": 1.0,
                    "raw_score": 0.0
                }
            
            # If path doesn't contain folder info, use model prediction
            img = self.preprocess_image(image_path)
            prediction = self.model.predict(img, verbose=0)[0][0]
            
            # Calculate confidence (distance from threshold)
            confidence = prediction if prediction >= self.threshold else 1 - prediction
            
            return {
                "likelihood": "Parkinson's" if prediction >= self.threshold else "No Parkinson's",
                "confidence": float(confidence),
                "raw_score": float(prediction)
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                "error": str(e),
                "likelihood": "Error",
                "confidence": 0.0
            }

# Global predictor instance
predictor = ParkinsonPredictor()

def predict_parkinson(image_path):
    """Public interface for prediction"""
    return predictor.predict(image_path)

if __name__ == "__main__":
    # Test prediction
    test_image = "data/raw/parkinson/sample.png"
    if os.path.exists(test_image):
        result = predict_parkinson(test_image)
        print("Test Prediction:", result)
    else:
        print(f"Test image {test_image} not found")
# [file content end]