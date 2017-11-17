from __future__ import print_function
import os
from hyperparams import Hyperparameters as hp
from random import shuffle
from PIL import Image
import numpy as np

class BatchData(object):
    def __init__(self, size, data_list):
        self._size = size
        self._data = data_list
    
    def get_data(self):
        images, labels = [], []
        for label, im_path in self._data:
            image = Image.open(im_path)
            image = image.convert('RGB')
            image = np.asarray(image, dtype=np.float)/255
            image = image[:, :, :3] #drop the alpha channel, not needed
            images.append(image)
            labels.append(label)
        return images, labels

class DataLoader(object):
    def __init__(self, batch_size=hp.batch_size):
        self.main_list = []
        self.data = []
        label = 0
        for dir in os.listdir(hp.train_dir):
            one_hot = [0, 0]
            one_hot[label] = 1
            for file in os.listdir(hp.train_dir+'/'+dir):
                self.main_list.append([one_hot, hp.train_dir+'/'+dir+'/'+file])
            label += 1
        shuffle(self.main_list)
        self.batch_size = batch_size
    
    def load_data(self):
        i = 0
        batch = []
        for label, im_path in self.main_list:
            batch.append([label, im_path])
            if len(batch) == self.batch_size:
                batch_data = BatchData(size=self.batch_size, data_list=batch)
                self.data.append(batch_data)
                batch=[]
        if len(batch) != 0:
            batch_data = BatchData(size=len(batch), data_list=batch)
            self.data.append(batch_data)
        return self.data