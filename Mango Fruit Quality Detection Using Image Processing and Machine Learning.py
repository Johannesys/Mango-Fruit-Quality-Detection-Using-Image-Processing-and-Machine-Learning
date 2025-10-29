

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from google.colab import drive, files

"""# Mango Quality Classification

This project implements a Convolutional Neural Network (CNN) to classify the quality of mangoes based on images. The model distinguishes between four stages of mango ripeness: Unripe, Early Ripe, Partially Ripe, and Ripe.

## Dataset

The dataset used for training and evaluation is located in the `/content/drive/My Drive/Implementasi Proyek/dataset/` directory and contains images categorized into the four ripeness stages.

## Model

The CNN model architecture consists of:
- Convolutional layers with ReLU activation.
- MaxPooling layers for down-sampling.
- A Flatten layer to convert the 3D feature maps to 1D.
- Dense layers with ReLU activation.
- A Dropout layer for regularization.
- An output Dense layer with Softmax activation for classification.

The model is compiled using the Adam optimizer and categorical crossentropy loss, with accuracy as the evaluation metric.

## Data Augmentation

To improve the model's robustness and generalization, data augmentation techniques such as rotation, shifting, shearing, zooming, and horizontal flipping are applied to the training data.

## Training

The model is trained using the augmented data with an early stopping callback to prevent overfitting. The training history, including accuracy and loss for both training and validation sets, is plotted to visualize the model's performance during training.

## Evaluation

The trained model is evaluated on a separate test set to assess its performance on unseen data. The test accuracy is reported.

## Usage

The trained model is saved as `mango_quality_model.h5`. A function is provided to load the saved model, preprocess a new image, and predict its quality stage. You can upload an image and the code will display the image with the predicted quality.

## How to Run

1. Clone this repository.
2. Ensure you have the required libraries installed (TensorFlow, Keras, OpenCV, NumPy, Pandas, Matplotlib, Seaborn).
3. Place your dataset in the specified directory structure on your Google Drive.
4. Run the Jupyter Notebook or Python script.
5. Use the provided code to load the model and make predictions on new images.
"""

# Load Data set

from google.colab import drive

drive.mount('/content/drive')

# Define the path to the dataset
dataset_path = '/content/drive/My Drive/Implementasi Proyek/dataset/'

filepath = []
labels = []

classlist = os.listdir(dataset_path)
for i in classlist:
    classpath = os.path.join(dataset_path, i)
    if os.path.isdir(classpath):
        flist = os.path.join(classpath)
        for f in flist:
            fpath = os.path.join(classpath, f)
            filepath.append(fpath)
            labels.append(i)


F = pd.Series(filepath, name = 'filepath')
L = pd.Series(labels, name = 'labels')
df = pd.concat([F, L], axis = 1)
categories = os.listdir(dataset_path)

print(df.head())
print(df['labels'].value_counts())

drive.mount('/content/drive')

# Define the path to the dataset
dataset_path = '/content/drive/My Drive/Implementasi Proyek/dataset/'

# List the categories (e.g., 'good', 'bad')
categories = os.listdir(dataset_path)
print(categories)

# Pre-Process Image
data = []
labels = []

# Define the image dimensions
IMG_SIZE = 128

for category in categories:
    path = os.path.join(dataset_path, category)
    class_num = categories.index(category)

    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img))
            resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append(resized_array)
            labels.append(class_num)
        except Exception as e:
            pass

# Convert lists to arrays
data = np.array(data)
labels = np.array(labels)

# Normalize image data
data = data / 255.0

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, len(categories))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size = 0.8, test_size=0.2, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train The Model
# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# Evaluate the Model
# Plotting training and validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()

# Evaluate on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

# Save The Model
model.save('mango_quality_model.h5')

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from google.colab import files
import matplotlib.pyplot as plt # Import matplotlib

# Define the image size and categories
IMG_SIZE = 128  # or the size your model expects
categories = ['Stage 0 (Unripe)', 'Stage 1 (Early Ripe)', 'Stage 2 (Partially Ripe)', 'Stage 3 (Ripe)']  # replace with your actual categories

# Load the model
model = load_model('mango_quality_model.h5')

# Function to preprocess and predict
def preprocess_and_predict(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for displaying with matplotlib
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return categories[np.argmax(prediction)]

def display_prediction(image_path, prediction):
    # Load the image using cv2 to maintain color consistency
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for displaying with matplotlib
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.title(f'Result: {prediction}', fontsize=15, fontweight='bold', loc='center', pad=20)
    plt.show()

# Test the function
uploaded = files.upload()
for filename in uploaded.keys():
    prediction = preprocess_and_predict(filename)
    display_prediction(filename, prediction)
