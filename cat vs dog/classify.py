import cv2
import os
import numpy as np
from random import shuffle
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression

classes=['dogs','cats']
num=len(classes)

training_path="train/"
validation_size=0.2
batch_size=16


def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat': return [1,0]
    #                             [no cat, very doggo]
    elif word_label == 'dog': return [0,1]

#creating data from source
def create_data():
    training_data = []
    for img in os.listdir(training_path):
        label = label_img(img)
        path = os.path.join(training_path,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (200,200))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

train_data=create_data()
MODEL_NAME = 'dogsvscats.model'.format(1e-3, '2conv-basic') 
#defining model
cn=input_data(shape=[None,200,200,1],name='input')

cn=conv_2d(cn,32,5,activation='relu')
cn=max_pool_2d(cn,5)

cn=conv_2d(cn,64,5,activation='relu')
cn=max_pool_2d(cn,5)

cn=conv_2d(cn,32,5,activation='relu')
cn=max_pool_2d(cn,5)

cn=conv_2d(cn,64,5,activation='relu')
cn=max_pool_2d(cn,5)

cn=conv_2d(cn,32,5,activation='relu')
cn=max_pool_2d(cn,5)

cn=conv_2d(cn,64,5,activation='relu')
cn=max_pool_2d(cn,5)

cn=fully_connected(cn,512,activation='relu')
cn=dropout(cn,0.8)

cn=fully_connected(cn,2,activation='softmax')

cn=regression(cn,optimizer='adam',learning_rate=1e-3,loss='categorical_crossentropy',name='targets')
model=tflearn.DNN(cn)

shuffle(train_data)
train_data=train_data[:2000]
train=train_data[:-150]
test=train_data[-150:]

X=np.array([i[0] for i in train]).reshape(-1,200,200,1)
Y=[i[1] for i in train]
X_test=np.array([i[0] for i in train]).reshape(-1,200,200,1)
Y_test=[i[1] for i in train]

model.fit({'input':X},{'targets':Y},n_epoch=10,validation_set=({'input':X_test},{'targets':Y_test}),snapshot_step=1000,show_metric=True)
x=test[0][0]
x=np.array(x).reshape(-1,200,200,1)
print type(x)
p=model.predict(x)
print (p)
model.save(MODEL_NAME)

