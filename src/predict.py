import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model('model.h5')

# Define image size
IMG_SIZE = (128, 128)

# Match the correct order from train_generator.class_indices
class_names = ['Potato___Early_blight', 'Tomato___Healthy', 'Tomato___Late_blight']

# Load and preprocess the image
img_path = 'test.jpg'  # Replace with your test image
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]

print(f"✅ Predicted disease: {predicted_class}")
