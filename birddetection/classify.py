# import the necessary packages
from hog import HOG
import detection
import argparse
import cPickle
import mahotas
import cv2
import cv2
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()


from sklearn.metrics import confusion_matrix
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True,
	help = "path to where the model will be stored")
ap.add_argument("-i", "--image", required = True,
	help = "path to the image file")
args = vars(ap.parse_args())

# load the model
model = open(args["model"]).read()
model = cPickle.loads(model)

# initialize the HOG descriptor
hog = HOG(orientations = 28, pixelsPerCell = (10, 10),
	cellsPerBlock = (1, 1), normalize = True)

# load the image and convert it to grayscale
image4 = cv2.imread(args["image"])
gray = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)
image = cv2.resize(gray, (47,62), interpolation = cv2.INTER_AREA)
# blur the image, find edges, and then find contours along
# the edged regions


blurred = cv2.GaussianBlur(image, (15, 15), 0)
edged = cv2.Canny(blurred, 30, 150)

cv2.imwrite("im.jpg",edged)
im=cv2.imread("im.jpg")
ss.setBaseImage(im)
ss.switchToSelectiveSearchQuality()
rects = ss.process()
par=0
print('Total Number of Region Proposals: {}'.format(len(rects)))
if(len(rects)<2):
	blurred = cv2.GaussianBlur(image, (5, 5), 0)
	edged = cv2.Canny(blurred, 30, 150)

	cv2.imwrite("im.jpg",edged)
	im=cv2.imread("im.jpg")
	ss.setBaseImage(im)
	par=1

	ss.switchToSelectiveSearchQuality()
	rects = ss.process()
rect2=[]

for r in rects:
	(x,y,w,h)=r
	rect2.append(r)
	if par==1:
		r2=(x+w/4,y+h/4,x+w/2,y+h/2)
		rect2.append(r2)
	


c=0
rect3=[]
# extract features from the image and classify it
for rect in rect2:
	
	(x,y,w,h)=rect
	if(w==0 or h==0 ):
		continue
	
	thresh=edged[y:y+h,x:x+w]
	#thresh = detection.deskew(edged, 20)
	thresh = detection.center_extent(thresh, (20, 20))
	hist = hog.describe(thresh)
#print (hist)
	hist=hist.reshape(1,-1)
#print (hist)
	digit = model.predict(hist)
        if (digit=="bird"):
	   rect3.append(rect)
	   print (rect)

rect4=[]

#for r in rect3:
#	(x,y,w,h)=r
#	cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 255), 1)
if(par==0):
	rect5=cv2.groupRectangles(rect3,1,0.5)
#print (rect5)
if(par==1):
	rect5=cv2.groupRectangles(rect3,1,1)
	print("yes")

for r in rect5[0]:
	(x,y,w,h)=r
	cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 1)


im0=cv2.resize(im,(400,400))
cv2.imshow("im",im0)
im1=cv2.resize(image4,(400,400))
cv2.imshow("im2",im1)
cv2.waitKey(0)
print (im.shape)

print (c)
