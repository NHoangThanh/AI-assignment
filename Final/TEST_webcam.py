import tensorflow as tf
import numpy as np
import cv2
import os
import cvlib as cv
                    
# load model
model = tf.keras.models.load_model('face_detection.h5')

# Mở webcam
webcam = cv2.VideoCapture(0)

classes = ['man','woman']

while webcam.isOpened():
    status, frame = webcam.read()
    face, confidence = cv.detect_face(frame)
    for idx, f in enumerate(face):

        # lấy điểm gốc của khung mặt        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        # vẽ hình vuông
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
        face_crop = np.copy(frame[startY:endY,startX:endX])
        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue
        # xử lý 
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = tf.keras.utils.img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
        # dự đoán khung mặt
        conf = model.predict(face_crop)[0] 
        idx = np.argmax(conf)
        label = classes[idx]
        label = "{}: {:.2f}%".format(label, conf[idx] * 100)
        Y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # display output
    cv2.imshow("FACE DETECTION", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()


