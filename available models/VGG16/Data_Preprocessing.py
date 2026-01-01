import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img
from sklearn.model_selection import train_test_split

data_dir = r"C:\Users\hp\Desktop\ML_Project\VGG16\Traffic_Labeled"
output_dir = r"C:\Users\hp\Desktop\ML_Project\VGG16\Traffic_Modified"

os.makedirs(output_dir, exist_ok=True)

datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,  # Random shear transformations
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'  # Strategy for filling in newly created pixels during transformations
)

# Get the class names (subdirectories in data_dir)
class_names = os.listdir(data_dir)

# Process each class and augment its images
for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        # Create a directory to save the preprocessed images and augmentations for this class
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        # List all the images in this class
        images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

        for image_name in images:
            # Load and preprocess the image
            image_path = os.path.join(class_dir, image_name)
            img = load_img(image_path, target_size=(224, 224))  # Resize to 224x224 (VGG16 input size)
            img_array = img_to_array(img)  # Convert image to array

            # Normalize the image
            img_array /= 255.0  # Normalize pixel values to [0, 1]

            # Save the preprocessed image (one copy per original image)
            preprocessed_path = os.path.join(class_output_dir, f"preprocessed_{image_name}")
            save_img(preprocessed_path, img_array)  # Save the preprocessed image

            # Augment and save 5 augmented versions of each image
            img_array = img_array.reshape((1,) + img_array.shape)  # Reshape for the ImageDataGenerator
            i = 0
            for batch in datagen.flow(img_array, batch_size=1, save_to_dir=class_output_dir, save_prefix='aug',
                                      save_format='jpeg'):
                i += 1
                if i >= 5:  # Generate 5 augmented images per original image
                    break

print("Preprocessing and augmentation complete!")
