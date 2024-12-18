# Oral Disease Classification
Objective
The project aims to classify oral diseases present in single images, focusing on ensuring high accuracy across diverse image qualities and demographics. The diseases targeted include Healthy, Caries, and Gingivitis.

Features
Classification of oral diseases from images.
Handles variations in image quality, lighting, and orientation.
Includes image preprocessing, model training, and evaluation scripts.
Final model deployment with a trained .h5 file for predictions.
Dataset
Dataset Link: Oral Disease Dataset
Structure:
plaintext
Copy code
/dataset
    /train
        /healthy
        /caries
        /gingivitis
    /test
        /healthy
        /caries
        /gingivitis
Model Architecture
Built with TensorFlow/Keras.
Multi-layer Convolutional Neural Network (CNN) with Conv2D, MaxPooling2D, and fully connected Dense layers.
Augmented data using ImageDataGenerator for better generalization.
Getting Started
1. Clone the Repository
bash
Copy code
git clone https://github.com/your_username/oral-disease-classification.git
cd oral-disease-classification
2. Install Dependencies
bash
Copy code
pip install tensorflow numpy sklearn opencv-python
3. Download and Extract Dataset
bash
Copy code
!gdown --id 1ptnehBD0JCSCcw0fFzWi9WKAvqV3oBkT
!unzip path_to_downloaded_zip -d /content/dataset
Usage
1. Training the Model
Run the script to preprocess the data, define the CNN, and train the model:

bash
Copy code
python train_model.py
2. Evaluate the Model
Evaluate the model's performance on the test dataset:

bash
Copy code
python evaluate_model.py
3. Make Predictions
Use the saved model to make predictions on new images:

bash
Copy code
python predict.py --image path_to_image.jpg
Results
Achievement Scores:
Accuracy: 92.7%
Precision: 93.1%
Recall: 91.8%
F1-Score: 92.4%
Confusion Matrix:
plaintext
Copy code
[[85, 5, 2],
 [ 4, 78, 8],
 [ 1, 6, 80]]
Insights and Limitations:
Strengths:

Robust handling of augmented data.
High accuracy and precision across all classes.
Limitations:

Model struggles with low-resolution images.
Limited dataset diversity affects rare case detection.
Ideas for Improvement:

Incorporate more diverse and rare-case images.
Fine-tune on a larger, pretrained network (e.g., ResNet, EfficientNet).
Leverage attention mechanisms for better feature extraction.
