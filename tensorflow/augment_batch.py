import random
from random import randint
import numpy as np
import cv2

class AugmentDrivingBatch:

    def __init__(self):
        # Initialize seed
        #random.seed(42)
        # Create a list of functions that could be applied on the batch
        self.__list_func = [lambda img: self.convert_to_gray(img), lambda img: self.add_noise(img),
                            lambda img: self.add_gaussian(img), lambda img: self.convert_to_sepia(img),
                            lambda img: self.color_swap(img),
                            lambda img: self.invert_color(img), lambda img: self.random_shadow(img)]

    def augment(self, batch):
        # Roll the dice
        prob = random.random()

        # Half chance of nothing half do some augmentation
        if prob < 0.5:
            return batch
        else:
            # Do a copy of the batch
            new_batch = batch

            # Flip steering independent of other augmentations (Idea is to have more steering actions on training)
            batch_fliped = self.create_flip_steering(new_batch)

            # Choose one operation to be applied on the whole batch
            #operation = randint(0, len(self.__list_func) - 1)

            # Do augmentations
            idx = 0
            for (img, steering) in batch_fliped:
                # Choose one operation to be applied on each image of the batch
                operation = randint(0, len(self.__list_func) - 1)
                # Choose the operation randomically
                img = self.__list_func[operation](img)
                batch_fliped[idx] = (img,steering)
                idx += 1

            return batch_fliped

    def convert_to_gray(self, img):
        # Get each channel
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # To keep same number of channels add gray to each one.
        img[:, :, 0] = gray
        img[:, :, 1] = gray
        img[:, :, 2] = gray
        return img

    def convert_to_sepia(self, img):
        # Get each channel
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        # To keep same number of channels add gray to each one.
        img[:, :, 0] = 0.393 * r + 0.769 * g + 0.189 * b
        img[:, :, 1] = 0.349 * r + 0.686 * g + 0.168 * b
        img[:, :, 2] = 0.272 * r + 0.534 * g + 0.131 * b
        return img

    def random_shadow(self, img):
        """
        Generates and adds random shadow
        """
        # (x1, y1) and (x2, y2) forms a line
        # xm, ym gives all the locations of the image
        image = (img * 255.0).astype('uint8')
        x1, y1 = image.shape[1] * np.random.rand(), 0
        x2, y2 = image.shape[1] * np.random.rand(), image.shape[0]
        xm, ym = np.mgrid[0:image.shape[0], 0:image.shape[1]]

        # mathematically speaking, we want to set 1 below the line and zero otherwise
        # Our coordinate is up side down.  So, the above the line:
        # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
        # as x2 == x1 causes zero-division problem, we'll write it in the below form:
        # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
        mask = np.zeros_like(image[:, :, 1])
        mask[np.where((ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0)] = 1

        # choose which side should have shadow and adjust saturation
        cond = mask == np.random.randint(2)
        s_ratio = np.random.uniform(low=0.2, high=0.5)

        # adjust Saturation in HLS(Hue, Light, Saturation)
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
        return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB).astype(img.dtype) / 255.0

    def add_noise(self, img):
        #new_img = skimage.util.random_noise(img,var=0.001)
        return img
        #return new_img

    def invert_color(self, img):
        new_img = img#np.invert(img)
        return new_img

    def add_gaussian(self, img):
        new_img = img#skimage.filters.gaussian(img,sigma=0.9, multichannel=True)
        return new_img

    def color_swap(self, img):
        new_img = img
        list_chanels = [0, 1, 2]
        random.shuffle(list_chanels)
        new_img[:, : ,0] = img[:, :, list_chanels[0]]
        new_img[:, :, 1] = img[:, :, list_chanels[1]]
        new_img[:, :, 2] = img[:, :, list_chanels[2]]
        return new_img

    # Flip both the image and the steering
    def create_flip_steering(self, batch):
        # Do a copy of the batch
        new_batch = batch
        idx = 0
        for (img, steering) in new_batch:
            if steering != 0.0:
                img = np.fliplr(img)
                # TODO: Why this is a list?
                steering[0] = -steering[0]
            new_batch[idx] = (img, steering)
            idx += 1
        return new_batch

    def display_batch(self, batch):
        pass
        #for img, steering in batch:
        #    plt.imshow(img)
        #    plt.show()
