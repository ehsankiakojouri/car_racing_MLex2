import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Function to load and preprocess images
def load_and_preprocess_images(directory, img_size=(224, 224)):
    images = []
    labels = []
    class_names = sorted(os.listdir(directory))

    for class_name in class_names:
        class_path = os.path.join(directory, class_name)
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
            images.append(img_array)
            labels.append(class_name)

    return np.array(images), np.array(labels)


# Load and preprocess training images
train_images, train_labels = load_and_preprocess_images('./train')

# Load and preprocess test images
test_images, test_labels = load_and_preprocess_images('./test')

# Save as NumPy arrays
np.save('train_images.npy', train_images)
np.save('train_labels.npy', train_labels)
np.save('test_images.npy', test_images)
np.save('test_labels.npy', test_labels)
