import os
import shutil
import random

source_dir = 'PlantVillage'
train_dir = 'dataset/train'
test_dir = 'dataset/test'
split_ratio = 0.8

for folder in os.listdir(source_dir):
    os.makedirs(os.path.join(train_dir, folder), exist_ok=True)
    os.makedirs(os.path.join(test_dir, folder), exist_ok=True)

    images = os.listdir(os.path.join(source_dir, folder))
    random.shuffle(images)

    split_point = int(len(images) * split_ratio)
    train_images = images[:split_point]
    test_images = images[split_point:]

    for img in train_images:
        src = os.path.join(source_dir, folder, img)
        dst = os.path.join(train_dir, folder, img)
        shutil.copyfile(src, dst)

    for img in test_images:
        src = os.path.join(source_dir, folder, img)
        dst = os.path.join(test_dir, folder, img)
        shutil.copyfile(src, dst)

print("✅ Dataset organized successfully!")
