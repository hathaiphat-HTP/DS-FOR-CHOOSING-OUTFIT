from flask import Flask, render_template, request
from datetime import datetime
import os

# pre processing
import cv2
import numpy as np

# deep learning model
import tensorflow as tf
MODEL = tf.keras.models.load_model('models/my_model.h5')

label_mapping = ['Apple', 'Hourglass', 'Pear', 'Rectangle']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# สร้างโฟลเดอร์สำหรับเก็บไฟล์ที่อัพโหลด ถ้ายังไม่มีโฟลเดอร์
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/', methods=['GET', 'POST'])
def index():
    result_label = None
    if request.method == 'POST':
        # รับข้อมูลจากฟอร์ม
        weight = request.form['weight']
        height = request.form['height']
        age = request.form['age']
        
        # รับไฟล์รูปภาพจากฟอร์ม
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        
        if file.filename == '':
            return "No selected file"
        
        if file:
            # ดึงเวลาปัจจุบัน
            now = datetime.now()

            # สร้างชื่อไฟล์ในรูปแบบที่ต้องการ
            filename = now.strftime("%y_%m_%d_%H_%M_%S") + ".png"
            # บันทึกไฟล์รูปภาพ
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # pre processing data
            data_attributes, data_image = pre_processing(weight, height, age, filepath)
            print(data_attributes, data_image)

            # predict model
            result = MODEL.predict([data_attributes, data_image])
            print(result)
            # Get the predicted class
            result_index = np.argmax(result, axis=1)[0]
            result_label = label_mapping[result_index]

            # print(f"Predicted Class Index: {result_index}")
            print(f"Predicted Class Label: {result_label}")

            # delete images
            os.remove(filepath)
    
    return render_template('index.html', result_label=result_label)


def pre_processing(weight, height, age, image_path):
    # โหลดรูปภาพจากเส้นทาง, แปลงขนาดเป็น 256x256 และแปลงเป็นภาพขาวดำ

    # Load the image
    image = cv2.imread(image_path)
    # Resize to 256x256
    image_resized = cv2.resize(image, (256, 256))

    # # Convert to grayscale
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    data_attributes = np.array([age, weight, height], dtype=np.float32).reshape(1, -1)
    data_image = np.array(image_gray, dtype=np.float32).reshape(1, 256, 256, 1)

    return data_attributes, data_image

if __name__ == '__main__':
    app.run(debug=True)
