# [file name]: train_model.py
# [file content begin]
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import numpy as np
import os
import logging
from preprocess import preprocess_images

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_class_weights(data_dir):
    """Calculate class weights to handle imbalanced datasets"""
    try:
        parkinson_dir = os.path.join(data_dir, 'parkinson')
        no_parkinson_dir = os.path.join(data_dir, 'no_parkinson')
        
        parkinson_count = len(os.listdir(parkinson_dir)) if os.path.exists(parkinson_dir) else 0
        no_parkinson_count = len(os.listdir(no_parkinson_dir)) if os.path.exists(no_parkinson_dir) else 0
        
        if parkinson_count == 0 or no_parkinson_count == 0:
            raise ValueError("One or both class directories are empty or don't contain images")
        
        total = parkinson_count + no_parkinson_count
        return {
            0: total / (2 * no_parkinson_count),  # No Parkinson's
            1: total / (2 * parkinson_count)      # Parkinson's
        }
    except Exception as e:
        logger.error(f"Error calculating class weights: {str(e)}")
        raise

def build_model(input_shape=(128, 128, 1)):
    """Build and compile the CNN model"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

def train_model():
    try:
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
        
        # Load and preprocess data
        train_data, val_data = preprocess_images("data/raw")
        
        # Calculate class weights
        class_weights = calculate_class_weights("data/raw")
        logger.info(f"Using class weights: {class_weights}")
        
        # Build model
        model = build_model()
        model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True),
            ModelCheckpoint("models/parkinson_model.keras", monitor='val_auc', save_best_only=True, mode='max')
        ]
        
        # Train model
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=50,
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        # Evaluate model
        y_true = np.concatenate([y for x, y in val_data], axis=0)
        y_pred = model.predict(val_data)
        
        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Save threshold
        with open("models/optimal_threshold.txt", "w") as f:
            f.write(str(optimal_threshold))
        
        logger.info(f"Optimal threshold: {optimal_threshold}")
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()
# [file content end]