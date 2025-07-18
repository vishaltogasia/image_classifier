from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ✅ Load the trained transfer learning model
model = load_model('dog_cat_model_pretrained.h5')

# ✅ Class labels for binary classification
class_names = ['Cat', 'Dog']

# ✅ Prepare image for prediction
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # normalization
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)

            # ✅ Predict the class
            img = prepare_image(filepath)
            pred = model.predict(img)[0][0]  # sigmoid output (scalar)

            # ✅ Class and confidence
            prediction = class_names[1] if pred > 0.5 else class_names[0]
            confidence = round(pred * 100, 2) if pred > 0.5 else round((1 - pred) * 100, 2)

            return render_template(
                'index.html',
                filename=filename,
                prediction=prediction,
                confidence=confidence
            )

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename))

if __name__ == '__main__':
    app.run(debug=True)
