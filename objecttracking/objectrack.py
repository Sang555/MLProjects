import cv2
import time
import argparse
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument("-v","--video",help="video")
args=vars(ap.parse_args())
bluelower=np.array([100,35,0],dtype=np.uint8)
blueupper=np.array([255,100,100],dtype=np.uint8)

if not args.get("video"):
	camera=cv2.VideoCapture(0)
else:
	camera=cv2.VideoCapture(args["video"])
while True:
	(grabbed,frame)=camera.read()
	if not grabbed:
		break
	
	frame2=cv2.inRange(frame,bluelower,blueupper)
	frame3=cv2.GaussianBlur(frame2,(3,3),0)
	(image,cnts,_)=cv2.findContours(frame3.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	if len(cnts)>0:
		cnts2=sorted(cnts,key=cv2.contourArea,reverse=True)[0]
	
		rect=np.int32(cv2.boxPoints(cv2.minAreaRect(cnts2)))
		cv2.drawContours(frame,[rect],-1,(0,255,0),2)
	cv2.imshow("frame",frame)
	time.sleep(0.025)
	if cv2.waitKey(1) & 0xFF==ord("q"):
		break
camera.release()
cv2.destroyAllWindows()
