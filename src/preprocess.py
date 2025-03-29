import tensorflow as tf
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_images(data_dir, img_size=(128, 128), batch_size=32):
    """Preprocess images and create train/validation datasets"""
    try:
        # Verify data directory structure
        parkinson_dir = os.path.join(data_dir, 'parkinson')
        no_parkinson_dir = os.path.join(data_dir, 'no_parkinson')
        
        if not os.path.exists(parkinson_dir) or not os.path.exists(no_parkinson_dir):
            raise ValueError("Data directory must contain 'parkinson' and 'no_parkinson' subdirectories")
        
        if len(os.listdir(parkinson_dir)) == 0 or len(os.listdir(no_parkinson_dir)) == 0:
            raise ValueError("One or both class directories are empty")
        
        # Load datasets
        train_data = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            image_size=img_size,
            batch_size=batch_size,
            color_mode="grayscale",
            label_mode="binary",
            validation_split=0.2,
            subset="training",
            seed=42
        )

        val_data = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            image_size=img_size,
            batch_size=batch_size,
            color_mode="grayscale",
            label_mode="binary",
            validation_split=0.2,
            subset="validation",
            seed=42
        )

        # Normalize and optimize datasets
        normalization = tf.keras.layers.Rescaling(1./255)
        train_data = train_data.map(lambda x, y: (normalization(x), y))
        val_data = val_data.map(lambda x, y: (normalization(x), y))
        
        # Optimize performance
        train_data = train_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        val_data = val_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        
        logger.info("Dataset preprocessed successfully!")
        return train_data, val_data
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    train_data, val_data = preprocess_images("data/raw")