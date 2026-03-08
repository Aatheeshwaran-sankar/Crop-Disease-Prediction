import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Set image size and paths
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

TRAIN_PATH = '../dataset/train'
TEST_PATH = '../dataset/test'

# Step 1: Load dataset using ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Step 2: Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(train_data.num_classes, activation='softmax')
])

# Step 3: Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Train the model
EPOCHS = 10
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data
)

# Step 5: Save the trained model
os.makedirs("model", exist_ok=True)
model.save("../model/crop_disease_model.h5")


print("✅ Model trained and saved Perfectly")


