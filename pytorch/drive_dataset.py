# References
# https://github.com/hminle/car-behavioral-cloning-with-pytorch/blob/master/utils.py
# https://github.com/hminle/car-behavioral-cloning-with-pytorch/blob/master/experiment.ipynb

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import scipy.misc
import lmdb

import random
from random import randint
from random import shuffle
import cv2


# Transform that augment the driving angles
class AugmentDrivingTransform(object):
    def __call__(self, sample):
        image = sample['image']
        steering = sample['label']
        # Only augment steering that is not zero
        if steering != 0:
            # Roll the dice
            prob = random.random()
            # Half chance of nothing half do some augmentation
            if prob > 0.5:
                # Flip image and steering angle
                sample['image'] = np.fliplr(sample['image'])
                sample['label'] = -steering

        return sample

class RandomBrightness(object):
    def __call__(self, sample):
        image = sample['image']
        steering = sample['label']        
        # Roll the dice
        prob = random.random()
        # Half chance of nothing half do some augmentation
        if prob > 0.5:            
            # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            ratio = 1.0 + 0.1 * (np.random.rand() - 0.5)
            hsv[:,:,2] =  hsv[:,:,2] * ratio
            sample['image'] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype('float32')

        return sample


class ConvertToGray(object):
    def __call__(self, sample):
        image = sample['image']
        steering = sample['label']        
        # Roll the dice
        prob = random.random()
        # Half chance of nothing half do some augmentation
        if prob > 0.5:            
            # Get each channel
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            # To keep same number of channels add gray to each one.
            img_new = image.copy()
            img_new[:, :, 0] = gray
            img_new[:, :, 1] = gray
            img_new[:, :, 2] = gray
            sample['image'] = img_new.astype('float32')

        return sample


class ConvertToSepia(object):
    def __call__(self, sample):
        image = sample['image']
        steering = sample['label']        
        # Roll the dice
        prob = random.random()
        # Half chance of nothing half do some augmentation
        if prob > 0.5:            
            # Get each channel
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            img_new = image.copy()
            img_new[:, :, 0] = 0.393 * r + 0.769 * g + 0.189 * b
            img_new[:, :, 1] = 0.349 * r + 0.686 * g + 0.168 * b
            img_new[:, :, 2] = 0.272 * r + 0.534 * g + 0.131 * b
            sample['image'] = img_new.astype('float32')

        return sample


class AddNoise(object):
    def __call__(self, sample):
        image = sample['image']
        steering = sample['label']        
        # Roll the dice
        prob = random.random()
        # Half chance of nothing half do some augmentation
        if prob > 0.5:            
            row,col,ch = image.shape
            gauss = np.random.randn(row,col,ch)
            gauss = gauss.reshape(row,col,ch) / 30.0
            img_new = image + image * gauss
            sample['image'] = img_new.astype('float32')

        return sample


class DrivingDataToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image.copy()), 'label': label}


class DriveData_LMDB(Dataset):
    __xs = []
    __ys = []
    __env = []

    def __init__(self, folder_dataset, transform=None):
        self.transform = transform
        # Load LMDB file
        print('Load LMDB:', folder_dataset)
        self.__env = lmdb.open(folder_dataset, readonly=True)

        # Open and load LMDB file including the whole training data (And load to memory)
        with self.__env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                key_str = key.decode('ascii')
                # print(key_str)
                if 'label' in key_str:
                    self.__ys.append(np.float32(np.asscalar(np.frombuffer(value, dtype=np.float32, count=1))))
                else:
                    # Get shape information from key name
                    info_key = key_str.split('_')
                    # Get image shape [2:None] means from index 2 to the end
                    shape_img = tuple(map(lambda x: int(x), info_key[2:None]))
                    # Convert to float32
                    self.__xs.append(np.frombuffer(value, dtype=np.uint8).reshape(shape_img))

    def addFolder(self, folder_dataset):
        print('Not supported now for LMDB')
        pass

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        img = self.__xs[index]
        img = (scipy.misc.imresize(img[126:226], [66, 200]) / 255.0).astype('float32')

        # Convert label to torch tensors
        label = self.__ys[index]

        # Do Transformations on the image/label
        sample = {'image': img, 'label': label}
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)