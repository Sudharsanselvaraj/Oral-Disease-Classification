# Import necessary libraries
import os
import shutil
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, classification_report

# Set image parameters
img_size = 224  # Target size for resizing images
batch_size = 32  # Batch size for training and testing

# Step 1: Dataset Setup and Loading
# Download dataset using gdown (make sure gdown is installed)
!gdown --id 1ptnehBD0JCSCcw0fFzWi9WKAvqV3oBkT

# Extract dataset to the desired location
!unzip path_to_downloaded_zip -d /content/dataset

# Verify the directory structure
for root, dirs, files in os.walk('/content/dataset'):
    print(root, dirs, files)

# Ensure directory structure is correct; reorganize if needed
os.makedirs('/content/dataset/train/healthy', exist_ok=True)
# Example for moving files to correct folders
# shutil.move('/path_to_images', '/content/dataset/train/healthy/')

# Step 2: Image Preprocessing
# Define data generators with augmentation for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    validation_split=0.2,  # Split 20% of training data for validation
    horizontal_flip=True,  # Randomly flip images horizontally
    rotation_range=20  # Random rotation for robustness
)

# Load training data
train_generator = train_datagen.flow_from_directory(
    '/content/dataset/train',  # Path to training data
    target_size=(img_size, img_size),  # Resize all images to the same size
    batch_size=batch_size,  # Batch size
    class_mode='categorical',  # Multi-class classification
    subset='training'  # Use the training split
)

# Load validation data
validation_generator = train_datagen.flow_from_directory(
    '/content/dataset/train',  # Path to training data
    target_size=(img_size, img_size),  # Resize all images
    batch_size=batch_size,  # Batch size
    class_mode='categorical',  # Multi-class classification
    subset='validation'  # Use the validation split
)

# Step 3: Model Training
# Define a CNN architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),  # Convolutional layer
    MaxPooling2D((2, 2)),  # Max pooling layer to reduce spatial dimensions

    Conv2D(64, (3, 3), activation='relu'),  # Another convolutional layer
    MaxPooling2D((2, 2)),  # Another max pooling layer

    Conv2D(128, (3, 3), activation='relu'),  # Deeper convolutional layer
    MaxPooling2D((2, 2)),  # Another max pooling layer

    Flatten(),  # Flatten feature maps into a 1D vector
    Dense(128, activation='relu'),  # Fully connected layer with 128 neurons
    Dropout(0.5),  # Dropout for regularization
    Dense(3, activation='softmax')  # Output layer for 3 classes
])

# Compile the model
model.compile(
    optimizer='adam',  # Adam optimizer
    loss='categorical_crossentropy',  # Loss function for multi-class classification
    metrics=['accuracy']  # Metric to track during training
)

# Train the model
history = model.fit(
    train_generator,  # Training data
    validation_data=validation_generator,  # Validation data
    epochs=10  # Number of training epochs
)

# Step 4: Model Evaluation
# Preprocess test data
test_datagen = ImageDataGenerator(rescale=1./255)  # Normalize test data
test_generator = test_datagen.flow_from_directory(
    '/content/dataset/test',  # Path to test data
    target_size=(img_size, img_size),  # Resize images
    batch_size=batch_size,  # Batch size
    class_mode='categorical'  # Multi-class classification
)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc}")

# Generate predictions on test data
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)  # Predicted class indices
y_true = test_generator.classes  # True class indices

# Calculate confusion matrix and classification report
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_true, y_pred))

# Step 5: Save Trained Model
# Save the model to disk
model.save('/content/oral_disease_model.h5')  # Save model as an H5 file

# Step 6: Test on a New Image
# Load and preprocess a sample image
sample_img_path = '/path_to_sample_image.jpg'
img = image.load_img(sample_img_path, target_size=(img_size, img_size))
img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input

# Predict the class of the sample image
prediction = model.predict(img_array)
class_index = np.argmax(prediction, axis=1)  # Get the class index
print(f"Predicted Class: {list(test_generator.class_indices.keys())[class_index[0]]}")

# End of Code
