from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import pandas as pd
import cv2
import numpy as np
import os
import time
from Web.services.Camera import VideoCamera

app = Flask(__name__)
app.config['SECRET_KEY'] = 'TryonGlasses'
socket = SocketIO(app)
myCamera = VideoCamera()

@app.route('/')
def main():
    df = pd.read_excel('static/Data.xlsx', sheet_name=0)
    df = df.to_dict('records')
    return render_template('index.html', data=df)

def gen(camera):
    while True:
        frame = camera.get_frame()[0]
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + frame +
               b'\r\n\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(gen(myCamera), mimetype="multipart/x-mixed-replace; boundary=frame")


@socket.on('chonKieng')
def chonKieng(IDKieng):
    myCamera.set_Glasses(IDKieng)
    return Response(gen(myCamera), mimetype="multipart/x-mixed-replace; boundary=frame")


@socket.on('deXuat')
def deXuat():
    gender = myCamera.get_frame()[1]
    age = myCamera.get_frame()[2]
    socket.emit('deXuat', (gender, age))


@socket.on('chup')
def chup():
    img = np.frombuffer(myCamera.get_frame()[0], dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)

    path = "C:/Users/USER/Pictures/Photo"

    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)

    timeName = time.strftime("%y-%m-%d_%H-%M-%S")
    imgName = path + '/' + timeName + '.png'

    cv2.imwrite(imgName, img)

    path = os.path.realpath(path)
    os.startfile(path)

if __name__ == "__main__":
    socket.run(app, allow_unsafe_werkzeug=True)
