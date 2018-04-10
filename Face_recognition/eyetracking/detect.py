import cv2
import argparse
from facedetection import Facedetector 

ap=argparse.ArgumentParser()
ap.add_argument("-f","--face",required=True,help="path to face cascade")
ap.add_argument("-e","--eye",required=True,help="path to eye cascade")
ap.add_argument("-i","--image",required=True,help="image")
args=vars(ap.parse_args())

image=cv2.imread(args["image"])
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
fd=Facedetector(args["face"],args["eye"])
rect=fd.detect(image,scaleFactor=1.2,minNeighbors=3,minSize=(30,30))
for (x,y,w,h) in rect:
	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow("detected",image)
cv2.waitKey(0)
