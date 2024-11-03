import os
import shutil
import random

# Set the path to your dataset and desired split ratio
dataset_path = "/users/jakelee/dataset"
train_path = os.path.join(dataset_path, "train")
validation_path = os.path.join(dataset_path, "validation")
split_ratio = 0.8  # 80% for training, 20% for validation

# Ensure the validation folders are created
os.makedirs(validation_path, exist_ok=True)

# Loop through each class and split images
for category in os.listdir(train_path):
    category_train_path = os.path.join(train_path, category)
    
    # Skip non-directory files like .DS_Store
    if not os.path.isdir(category_train_path):
        continue
    
    category_validation_path = os.path.join(validation_path, category)
    os.makedirs(category_validation_path, exist_ok=True)
    
    images = os.listdir(category_train_path)
    random.shuffle(images)
    
    # Split images
    split_point = int(len(images) * split_ratio)
    train_images = images[:split_point]
    validation_images = images[split_point:]
    
    # Move validation images
    for image in validation_images:
        shutil.move(os.path.join(category_train_path, image), category_validation_path)
