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