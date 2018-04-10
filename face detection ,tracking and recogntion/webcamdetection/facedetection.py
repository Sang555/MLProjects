import cv2
class Facedetector:
	def __init__(self,path):
		self.facecascade=cv2.CascadeClassifier(path)
	def detect(self,image,scaleFactor=1.1,minNeighbors=5,minSize=(30,30)):
		rects=self.facecascade.detectMultiScale(image,scaleFactor=scaleFactor,minNeighbors=minNeighbors,minSize=minSize,flags=cv2.cv2.CASCADE_SCALE_IMAGE)
		return rects
