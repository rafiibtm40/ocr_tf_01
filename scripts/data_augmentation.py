import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt

def create_data_augmentation_generator(X_train):
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Assuming X_train is an array of shape (num_samples, img_height, img_width, num_channels)
    # Example: Generating 5 augmented images from the first image in X_train
    augmented_images = datagen.flow(np.expand_dims(X_train[0], axis=0), batch_size=1)

    # Display a few augmented images to verify
    fig, ax = plt.subplots(1, 5, figsize=(15, 5))
    for i in range(5):
        ax[i].imshow(augmented_images.next()[0].astype('uint8'))
        ax[i].axis('off')
    plt.show()

# Example usage (pass in your actual training images)
# X_train = your_training_data
# create_data_augmentation_generator(X_train)
