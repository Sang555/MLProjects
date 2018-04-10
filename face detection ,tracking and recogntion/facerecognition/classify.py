# import the necessary packages
from hog import HOG
import detection
import argparse
import cPickle
import mahotas
import cv2
from sklearn.metrics import confusion_matrix
from facedetection import Facedetector 
face_cascade = cv2.CascadeClassifier('/home/sangee/opencv-3.3.0/data/haarcascade_frontalface_default.xml')
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True,
	help = "path to where the model will be stored")
ap.add_argument("-i", "--image", required = True,
	help = "path to the image file")
ap.add_argument("-f","--face",required=True,help="path to face cascade")
args = vars(ap.parse_args())

# load the model
model = open(args["model"]).read()
model = cPickle.loads(model)

# initialize the HOG descriptor
hog = HOG(orientations = 40, pixelsPerCell = (10, 10),
	cellsPerBlock = (1, 1), normalize = True)

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fd=Facedetector(args["face"])
faces=fd.detect(gray,scaleFactor=1.2,minNeighbors=3,minSize=(30,30))
#cv2.imshow('img',gray)
if (len(faces)==0):
	image = cv2.resize(image, (47,62), interpolation = cv2.INTER_AREA)
else:
	(x,y,w,h)=faces[0]
	roi_color = image[y:y+h, x:x+w]
	image = cv2.resize(roi_color, (47,62), interpolation = cv2.INTER_AREA)


# blur the image, find edges, and then find contours along
# the edged regions
blurred = cv2.GaussianBlur(image, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)

thresh = detection.deskew(edged, 20)
thresh = detection.center_extent(thresh, (20, 20))
# extract features from the image and classify it
hist = hog.describe(thresh)
#print (hist)
hist=hist.reshape(1,-1)
print (hist)
person = model.predict(hist)

print "The person is is: %s" % (person)

cv2.imshow("image", image)
		
cv2.waitKey(0)
