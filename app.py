from flask import Flask, render_template, request, redirect ,url_for
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
    # ตั้งค่าเริ่มต้นทุกครั้งที่ฟังก์ชันถูกเรียกใช้งาน
    result_label = None
    result_image = url_for('static', filename='default.jpg')  # ตั้งค่า default.jpg ทุกครั้งที่โหลดหน้าเว็บ

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
            
            # ทำการประมวลผลข้อมูลก่อนที่จะส่งไปยังโมเดล โดยใช้ฟังก์ชัน pre_processing
            data_attributes, data_image = pre_processing(weight, height, age, filepath)
            print(data_attributes, data_image)

            # ใช้โมเดล Deep Learning เพื่อทำนายรูปทรงของร่างกาย โดยส่งข้อมูลที่ประมวลผลแล้วเข้าไปในโมเดล
            result = MODEL.predict([data_attributes, data_image])
            print(result)
            # Get the predicted class
            result_index = np.argmax(result, axis=1)[0]
            result_label = label_mapping[result_index]

            # print(f"Predicted Class Index: {result_index}")
            print(f"Predicted Class Label: {result_label}")

            # ตรวจสอบเงื่อนไขตามผลลัพธ์ที่ได้จากโมเดล และแสดงรูปภาพที่เหมาะสม
            if result_label == 'Apple':
               result_image = url_for('static', filename='apple.jpg')
            elif result_label == 'Pear':
                result_image = url_for('static', filename='pear.jpg')
            elif result_label == 'Hourglass':
                result_image = url_for('static', filename='hourglass.jpg')
            elif result_label == 'Rectangle':
                result_image = url_for('static', filename='rectangle.jpg')
            else:
                result_image = url_for('static', filename='default.jpg') #รูปภาพเริ่มต้นกรณีที่ไม่มีเงื่อนไขตรงกัน

            # delete images
            os.remove(filepath)
            
    # คืนค่า result_label และ result_image ไปยังหน้าเว็บ
    return render_template('index.html', result_label=result_label, result_image=result_image)



def pre_processing(weight, height, age, image_path):
    # โหลดรูปภาพจากเส้นทาง, แปลงขนาดเป็น 256x256 และแปลงเป็นภาพขาวดำ

    # โหลดรูปภาพจากไฟล์
    image = cv2.imread(image_path)
    # ปรับขนาดของรูปภาพเป็น 256x256 พิกเซล
    image_resized = cv2.resize(image, (256, 256))

    # แปลงรูปภาพเป็นภาพขาวดำ grayscale
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # สร้างอาร์เรย์ที่เก็บข้อมูลอายุ น้ำหนัก และส่วนสูงที่ได้รับจากฟอร์ม และจัดรูปแบบให้เหมาะสมสำหรับการส่งให้โมเดล
    data_attributes = np.array([age, weight, height], dtype=np.float32).reshape(1, -1)
    data_image = np.array(image_gray, dtype=np.float32).reshape(1, 256, 256, 1)

    return data_attributes, data_image



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
