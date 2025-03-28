import numpy as np
import cv2
from tensorflow.keras.models import load_model

def preprocess_image(image_path, img_size=(128, 128)):
    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    return img

def predict_parkinson(image_path):
    # Load model
    model = load_model("models/parkinson_model.h5")

    # Load optimal threshold
    with open("models/optimal_threshold.txt", "r") as f:
        optimal_threshold = float(f.read())

    # Preprocess image
    img = preprocess_image(image_path)

    # Make prediction
    prediction = model.predict(img)[0][0]
    print(f"Raw prediction score: {prediction}")  # Debugging

    return "Parkinson's" if prediction >= optimal_threshold else "No Parkinson's"

# Example usage
if __name__ == "__main__":
    result = predict_parkinson("data/raw/parkinson/sample.png")
    print(f"Prediction result: {result}")