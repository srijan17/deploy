import flask
import cv2
import flask_cors
from flask import jsonify,request 
from flask_cors import CORS
from azure.storage.blob import BlockBlobService, PublicAccess
import os
import numpy as np

import urllib


app = flask.Flask(__name__)

@app.route('/test',methods=["POST"])
def test():
    
    url = 'http://dronefunctions96d0.blob.core.windows.net/box/'+request.json["ImageURL"]
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(image,cv2.IMREAD_COLOR)
    if(len(request.json["Barcode"])>=2):
        ymin=img.shape[0]
        ymax=0
        shelfValue=""
    

        for barcode in request.json["Barcode"]:
            if(barcode["Corner"]["bottom"]<ymin):
                ymin = barcode["Corner"]["bottom"]
            if(barcode["Corner"]["top"]>ymax):
                ymax = barcode["Corner"]["top"]
                shelfValue= barcode["Value"]
        imghsv = cv2.cvtColor(img[ymin:ymax,:,:],cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(imghsv, (10, 40, 20), (20, 255,255))

        
        ddepth = cv2.CV_32F
        gradX = cv2.Sobel(mask1, ddepth=ddepth, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(mask1, ddepth=ddepth, dx=0, dy=1, ksize=-1)
        
        # subtract the y-gradient from the x-gradient
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        blurred = cv2.blur(gradient, (5, 5))
        (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        gradient = cv2.morphologyEx(gradient, cv2.MORPH_DILATE, kernel, anchor=(2, 0), iterations=5) #dilatazione


        final = cv2.subtract(mask1,gradient)


        kernel = np.ones((5, 5), np.uint8)
        final = cv2.morphologyEx(final, cv2.MORPH_ERODE, kernel, anchor=(2, 0), iterations=5) #dilatazione
        cnts = cv2.findContours(final.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]
        count = 0
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if(area>100):
                count=count+1
        final= cv2.resize(final, (1000, 500)) 
        final2= cv2.resize(gradient, (1000, 500)) 
        print(count)

        response =  flask.jsonify({"box-count":count})
        response.headers.add('Access-Control-Allow-Origin', '*')

        # cv2.imshow(shelfValue,img[ymin:ymax])
        # cv2.imshow(shelfValue+"-erode",final)
        # cv2.imshow(shelfValue+"-hsv",imghsv)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
    return response

@app.route('/')
def hello_world():
  return 'Hey its Python Flask application!'



# In[ ]:

if __name__ == '__main__':
    app.run()
    CORS(app)

