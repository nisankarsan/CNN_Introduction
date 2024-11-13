# Cat vs Dog Classification using Convolutional Neural Networks

A deep learning project that classifies images of cats and dogs using CNN architecture built with TensorFlow and Keras. This implementation includes data preprocessing, model training, and evaluation capabilities.

## Dataset and Model Overview

### Dataset Specifications
• Source: Kaggle Dogs vs Cats dataset \
• Total Images: 25,000\
• Training Set: 20,000 images\
• Test Set: 5,000 images\
• Classes: Binary (Cats: 0, Dogs: 1)\
• Image Format: JPG\
• Input Size: 64x64x3 (RGB)

### Model Architecture
```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Install required packages
pip install tensorflow numpy kagglehub pillow matplotlib

# Configure Kaggle credentials
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

import kagglehub
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Download dataset
path = kagglehub.dataset_download("tongpython/cat-and-dog")

# Create data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

training_set = train_datagen.flow_from_directory(
    path + '/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
) 
```
I created cat-and-dog/single_prediction subfile for testing the my algorithm.
