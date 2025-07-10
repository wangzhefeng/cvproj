# -*- coding: utf-8 -*-

# ***************************************************
# * File        : app.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-10-31
# * Version     : 1.0.103100
# * Description : description
# * Link        : https://github.com/DharmarajPi/Opencv-face-detection-deployment-using-flask-API
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from flask import Flask, render_template, Response
import cv2

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]

app = Flask(__name__)


def capture_by_frames():
    global camera
    camera = cv2.VideoCapture(0)
    while True:
        # read the camera frame
        success, frame = camera.read()
        detector = cv2.CascadeClassifier("Haarcascades/haarcascade_frontalface_default.xml")
        faces = detector.detectMultiScale(frame, 1, 2, 6)
        # draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield(b"--frame\r\n"
              b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods = ["POST"])
def start():
    return render_template("index.html")

@app.route("/stop", methods = ["POST"])
def stop():
    if camera.isOpened():
        camera.release()
    return render_template("stop.html")

@app.route("./video_capture")
def video_capture():
    return Response(capture_by_frames(), mimetype = "multipart/x-mixed-replace; boundary=frame")




# 测试代码 main 函数
def main():
    app.run(debug = True, use_reloader = False, port = 8000)

if __name__ == "__main__":
    main()