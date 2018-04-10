import cv2
import argparse
from facedetection import Facedetector 

ap=argparse.ArgumentParser()
ap.add_argument("-f","--face",required=True,help="path to face cascade")
ap.add_argument("-e","--eye",required=True,help="path to eye cascade")
ap.add_argument("-v","--video",help="video")
args=vars(ap.parse_args())
fd=Facedetector(args["face"],args["eye"])
if not args.get("video"):
	camera=cv2.VideoCapture(0)
else:
	camera=cv2.VideoCapture(args["video"])
while True:		
	(grabbed, frame)= camera.read()
	if args.get("video") and not grabbed:
		break
	frame2=cv2.resize(frame,(300,300),cv2.INTER_AREA)
	frame3=cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
	rects=fd.detect(frame3,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
	for (x,y,w,h) in rects:
		cv2.rectangle(frame2,(x,y),(x+w,y+h),(0,255,0),2)
	cv2.imshow("detected",frame2)
	if cv2.waitKey(1) & 0xFF==ord("q"):
		break
camera.release()
cv2.destroyAllWindows()
