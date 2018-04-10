import cv2
class Facedetector:
	def __init__(self,path,eyepath):
		self.facecascade=cv2.CascadeClassifier(path)
		self.eyecascade=cv2.CascadeClassifier(eyepath)
	def detect(self,image,scaleFactor=1.1,minNeighbors=5,minSize=(30,30)):
		facerects=self.facecascade.detectMultiScale(image,scaleFactor=scaleFactor,minNeighbors=minNeighbors,minSize=minSize,flags=cv2.cv2.CASCADE_SCALE_IMAGE)
		rects=[]
		for (x,y,w,h) in facerects:
			rects.append((x,y,w,h))
	#cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
			roi=image[y:y+h,x:x+w]
			rects2=self.eyecascade.detectMultiScale(image,scaleFactor=1.2,minNeighbors=5,minSize=(20,20))	
			for (x2,y2,w2,h2) in rects2:
				rects.append((x2,y2,w2,h2))
		#cv2.rectangle(image,(x2,y2),(x2+w2,y2+h2),(0,255,0),2)
		return rects
