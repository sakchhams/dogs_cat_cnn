from __future__ import print_function
import os
from hyperparams import Hyperparameters as hp
import cv2

i = 0
data = os.listdir('train')
for dir in data:
    files = os.listdir('train/'+dir)
    for file in files:
        image = cv2.imread('train/'+dir+'/'+file)
        resized = cv2.resize(image, (hp.img_w, hp.img_h), interpolation = cv2.INTER_AREA)
        cv2.imwrite('train_01/'+dir+'/'+file, resized)
        i += 1
        print('dataset processing : {}'.format(i/25000))
