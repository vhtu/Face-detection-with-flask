#our web app framework!

#you could also generate a skeleton from scratch via
#http://flask-appbuilder.readthedocs.io/en/latest/installation.html

#Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
#HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine 
#for you automatically.
#requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template,request
from flask_cors import CORS, cross_origin
#scientific computing library for saving, reading, and resizing images
from scipy.misc import imsave, imread, imresize
#for matrix math
import numpy as np
#for importing our keras model
import keras.models
#for regular expressions, saves time dealing with string data
import re
import numpy as np
import cv2
import base64
#system level operations (like loading files)
import sys 
#for reading operating system data
import os
#initalize our flask app
app = Flask(__name__)


# Khai bao cong cua server
my_port = '8000'
scale = 0.00392
conf_threshold = 0.5
nms_threshold = 0.4

# load the face detector and detect faces in the image
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# Cac ham ho tro chay YOLO

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def build_return(class_id, x, y, x_plus_w, y_plus_h):
    return str(class_id) + "," + str(x) + "," + str(y) + "," + str(x_plus_w) + "," + str(y_plus_h)


# Khoi tao model YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

def convertImage1(imgData1):
    imgstr = re.search(b'base64,(.*)',imgData1).group(1)
    #print(imgstr) 
    with open('output.png','wb') as output: output.write(base64.b64decode(imgstr))



@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
	staff_id = request.args.get('id')
	return "Xin chao = "+str(staff_id)
	

@app.route('/predictPost/',methods=['GET','POST'])
def predictPost():
	staff_id = request.get_data()
	return "Xin chao = "+str(staff_id)


# Khai bao ham xu ly request detect
@app.route('/detect/', methods=['POST'])
@cross_origin()
def detect():
    # Lay du lieu image B64 gui len va chuyen thanh image
    image_b64 = request.get_data()
    convertImage1(image_b64)
    image = cv2.imread("output.png")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Nhan dien bang YOLO
    rects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=9,
    minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    # loop over the faces and draw a rectangle surrounding each
    str_rec = ''
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        str_rec = str_rec + str(x)+str(y)+str(w)+str(h)
    
    cv2.imwrite('ketqua.png',image)
    with open("ketqua.png", "rb") as img_file:
        my_string = base64.b64encode(img_file.read()).decode('utf-8')

    return str('data:image/png;base64,')+str(my_string);


if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True)
