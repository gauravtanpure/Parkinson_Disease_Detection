import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

def preprocess_images(data_dir, img_size=(128, 128), batch_size=32):
    # Load dataset
    train_data = image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        color_mode="grayscale",
        label_mode="int",
        validation_split=0.2,
        subset="training",
        seed=42
    )

    val_data = image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        color_mode="grayscale",
        label_mode="int",
        validation_split=0.2,
        subset="validation",
        seed=42
    )

    # Normalize images
    train_data = train_data.map(lambda x, y: (x / 255.0, y))
    val_data = val_data.map(lambda x, y: (x / 255.0, y))

    return train_data, val_data

# Example usage
if __name__ == "__main__":
    train_data, val_data = preprocess_images("data/raw")
    print("Dataset preprocessed successfully!")