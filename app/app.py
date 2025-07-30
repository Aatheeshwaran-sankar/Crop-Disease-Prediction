from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__, template_folder='../templates')

model = tf.keras.models.load_model("../model/crop_disease_model.h5")
class_names = ['Pepper-bell_Bacterial_spot', 'Pepper_bell_Healthy']  # Make sure this order matches your training

IMG_SIZE = (128, 128)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=IMG_SIZE)
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            result = class_names[np.argmax(prediction)]

            return render_template("index.html", prediction=result, image_path=filepath)

    return render_template("index.html", prediction=None)


if __name__ == "__main__":
    app.run(debug=True)

"""


from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__, template_folder='../templates')

# Load the trained model
#model = tf.keras.models.load_model('model/crop_disease_model.h5')
model = tf.keras.models.load_model('../model/crop_disease_model.h5')


# Define class names (update this list to match your training labels exactly)
class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy']

# Prediction route
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            img_path = os.path.join('static', file.filename)
            file.save(img_path)

            img = image.load_img(img_path, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            preds = model.predict(img_array)
            predicted_class = class_names[np.argmax(preds)]

            
            print("Predicted Disease:", predicted_class)

            prediction = predicted_class

    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
"""