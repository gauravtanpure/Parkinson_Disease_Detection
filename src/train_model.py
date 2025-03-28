import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import numpy as np
import os

# Load preprocessed dataset
from preprocess import preprocess_images

train_data, val_data = preprocess_images("data/raw")

# Calculate class weights
class_counts = np.array([len(os.listdir("data/raw/parkinson")), len(os.listdir("data/raw/no_parkinson"))])
total_samples = class_counts.sum()
class_weights = {0: total_samples / (2 * class_counts[0]), 1: total_samples / (2 * class_counts[1])}

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Add early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=[early_stopping],
    class_weight=class_weights
)

# Evaluate model
y_true = np.concatenate([y for x, y in val_data], axis=0)
y_pred = model.predict(val_data)

# Find optimal threshold using ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold}")

# Save optimal threshold to a file
with open("models/optimal_threshold.txt", "w") as f:
    f.write(str(optimal_threshold))

# Confusion matrix
y_pred_class = (y_pred > optimal_threshold).astype(int)
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_class))

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred_class))

# Save model
model.save("models/parkinson_model.h5")
print("Model saved successfully!")