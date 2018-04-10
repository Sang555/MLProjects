# import the necessary packages
from sklearn.svm import LinearSVC
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from hog import HOG
import detection
import argparse
import cPickle
import os
import numpy as np
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True,
	help = "path to where the model will be stored")
args = vars(ap.parse_args())

# load the dataset and initialize the data matrix

data = []
target=[]

for imgfolder in os.listdir('images/'):
	target.append("bird")

for imgfolder in os.listdir('noimages/'):
	target.append("nobird")
target = np.asarray(target)
print (target)

for imgfolder in os.listdir('images/'):
	filename='images/'+imgfolder
	image=cv2.imread(filename)
		
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
	image = cv2.resize(gray, (47,62), interpolation = cv2.INTER_AREA)
		
        data.append(image)
for imgfolder in os.listdir('noimages/'):
	filename='noimages/'+imgfolder
	image=cv2.imread(filename)
		
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
	image = cv2.resize(gray, (47,62), interpolation = cv2.INTER_AREA)
		
        data.append(image)
data=np.asarray(data)
print (data.shape)
data2=[]		
hog = HOG(orientations = 28, pixelsPerCell = (10, 10),
	cellsPerBlock = (1, 1), normalize = True)
for image2 in data:
	# deskew the image, center it
		
	#image = detection.deskew(image, 20)
	image = detection.center_extent(image, (20, 20))

	# describe the image and update the data matrix
	hist = hog.describe(image)
	data2.append(hist)

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
model = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid)

model.fit(data2, target)

# dump the model to file
f = open(args["model"], "w")
f.write(cPickle.dumps(model))
f.close()

data = np.asarray(data)
print (data.shape)
# initialize the HOG descriptor

