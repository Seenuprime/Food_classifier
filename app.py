from flask import Flask, render_template, request 
import tensorflow as tf
import matplotlib.image as mpimg
import os

app = Flask(__name__)
model = tf.keras.models.load_model('models/food_classifier_model.keras')

def preprocess_image(image_path):
    img = mpimg.imread(image_path)
    img = tf.image.resize(img, [336, 336])
    return img

def predict_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(tf.expand_dims(processed_image, axis=0)).argmax()
    class_names = ['Burger', 'Butter_naan', 'Chai', 'Chapati', 'Chole_bhature', 'Dal_makhani', 'Dhokla', 'Fried_rice', 'Idli', 'Jalebi', 'Kaathi_rolls', 'Kadai_paneer', 'Kulfi', 'Masala_dosa', 'Momos', 'Paani_puri', 'Pakode', 'Pav_bhaji', 'Pizza', 'Samosa']
    predicted_label = class_names[prediction]
    return prediction, predicted_label

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/reset', methods=['POST'])
def reset():
    uploads_dir = os.path.join(app.root_path, 'static', 'uploads')
    for filename in os.listdir(uploads_dir):
        file_path = os.path.join(uploads_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(e)
    return "Files deleted successfully"

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filename = file.filename
        uploads_dir = os.path.join(app.root_path, 'static', 'uploads')
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        file_path = os.path.join(uploads_dir, filename)
        file.save(file_path)
        prediction, predicted_label = predict_image(file_path)
        image_path = os.path.join('uploads', filename).replace("\\", "/")  # Replace backslashes with forward slashes
        return render_template('index.html', image_path=image_path, predicted_label=predicted_label)


if __name__ == '__main__':
    app.run(debug=True)
