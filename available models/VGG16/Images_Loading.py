import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Define the dataset path
data_dir = r"C:\Users\hp\Desktop\ML_Project\VGG16\Traffic_Modified"  # Replace with the actual path to your folder

# Initialize the ImageDataGenerator to load and preprocess images
datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values

# Load images and their labels using flow_from_directory
flow = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # For multi-class classification
    shuffle=False  # Don't shuffle so we can track the labels easily
)

# Get the labels and image paths
labels = flow.classes
image_paths = flow.filenames

# Encode the labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_encoded = to_categorical(labels_encoded)

# Get the number of samples (images)
num_samples = len(image_paths)

# Now, let's load all images into memory (be cautious if you have a large dataset)
images = np.zeros((num_samples, 224, 224, 3), dtype=np.float32)

# Manually load the images into the array
for i, image_path in enumerate(image_paths):
    img = load_img(os.path.join(data_dir, image_path), target_size=(224, 224))  # Correct method to load image
    img_array = img_to_array(img)  # Convert image to array
    images[i] = img_array

# Print dataset shape for debugging
print(f"Images Shape: {images.shape}")

# Now you can proceed with Stratified Cross-Validation as before.
