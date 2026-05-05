import os
import random
# import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img

# paths
DATASET_DIR = "/Users/jennyngo/Downloads/Test_Alphabet"
OUTPUT_DIR = "/Users/jennyngo/Documents/GitHub/asl-translator/augmented_samples"
IMG_SIZE = (224, 224)
SAMPLES_PER_CLASS = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

for class_name in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    # create class folder in output
    output_class_path = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    images = os.listdir(class_path)

    # pick 4 random images
    sampled_images = random.sample(images, min(SAMPLES_PER_CLASS, len(images)))

    for img_name in sampled_images:
        img_path = os.path.join(class_path, img_name)

        # load image
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)

        # expand dims
        img_array = np.expand_dims(img_array, axis=0)

        # apply augmentation
        aug_iter = datagen.flow(img_array, batch_size=1)
        aug_img = next(aug_iter)[0]

        # convert back to 0–255 for saving
        aug_img_uint8 = (aug_img * 255).astype(np.uint8)

        # save
        save_path = os.path.join(
            output_class_path,
            f"aug_{img_name}"
        )
        save_img(save_path, aug_img_uint8)

print("data augmentation saved")