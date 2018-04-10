import os
import cv2
import numpy as np
f = open('bounding_boxes.txt')
line = f.readline()
i=0

while (i<300):
    c=[]
    i=i+1
    for word in line.split():
           c.append(word)
    
    im=cv2.imread("train/"+c[0])
    im2=im[int(c[2]):int(c[2])+int(c[4]),int(c[1]):int(c[1]+c[3])]
    print (i)
    im4=im[0:int(c[2]),0:int(c[1])]
    im4=cv2.resize(im4,(200,200),interpolation = cv2.INTER_AREA)
    im5=im[int(c[2])+int(c[4]):np.size(im,0),0:int(c[1])]
    im5=cv2.resize(im5,(200,200),interpolation = cv2.INTER_AREA)
    im6=im[0:int(c[2]),int(c[1])+int(c[3]):np.size(im,1)]
    im6=cv2.resize(im6,(200,200),interpolation = cv2.INTER_AREA)
    im7=im[int(c[2])+int(c[4]):np.size(im,0),0:int(c[1])]
    im7=cv2.resize(im5,(200,200),interpolation = cv2.INTER_AREA)
    im8=im[0:int(c[2]),int(c[1]):int(c[1])+int(c[3])]
    im8=cv2.resize(im8,(200,200),interpolation = cv2.INTER_AREA)
    im3=cv2.resize(im2,(200,200),interpolation = cv2.INTER_AREA)
    cv2.imwrite("noimages/"+c[0],im4)
    cv2.imwrite("noimages/"+"1"+c[0],im5)
    cv2.imwrite("noimages/"+"2"+c[0],im6)
    cv2.imwrite("noimages/"+"3"+c[0],im7)
    cv2.imwrite("noimages/"+"4"+c[0],im8)
    cv2.imwrite( "images/"+c[0], im3)
    line = f.readline()

