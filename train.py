import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train_model():
    # Image dimensions
    img_width, img_height = 224, 224
    batch_size = 32
    epochs = 10

    # Dataset directories (ensure correct relative paths)
    train_dir = 'train'
    val_dir = 'val'
    test_dir = 'test'

    # Data preprocessing and augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    test_val_datagen = ImageDataGenerator(rescale=1.0/255)

    # Data generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    val_generator = test_val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    test_generator = test_val_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    # CNN Model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )

    # Evaluate model
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # Predictions
    Y_pred = model.predict(test_generator)
    y_pred = (Y_pred > 0.5).astype(int)
    y_true = test_generator.classes

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Save model
    model.save("bone_fracture_model.h5")
    print("Model saved as 'bone_fracture_model.h5'")

if __name__ == "__main__":
    train_model()
