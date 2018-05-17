import scipy.misc
import random
import h5py
import lmdb
import numpy as np
import tensorflow as tf
from augment_batch import AugmentDrivingBatch


class HandleData:
    __xs = []
    __ys = []
    __file = []
    __file_val = []
    __dataset_imgs = []
    __dataset_label = []
    __dataset_imgs_val = []
    __dataset_label_val = []
    __num_images = 0
    __train_xs = []
    __train_ys = []
    __val_xs = []
    __val_ys = []
    __num_train_images = 0
    __num_val_images = 0
    __train_batch_pointer = 0
    __val_batch_pointer = 0
    __train_perc = 0
    __val_perc = 0
    __augment = None
    __split_training = False

    # points to the end of the last batch
    __train_batch_pointer = 0
    __val_batch_pointer = 0

    def __init__(self, path='DrivingData.h5', path_val='', train_perc=0.8, val_perc=0.2, shuffle=True):
        self.__augment = AugmentDrivingBatch()
        self.__train_perc = train_perc
        self.__val_perc = val_perc
        print("Loading training data")
        # Handle HDF5/LMDB datasets (Load content to memory)
        self.handle_file_dataset(path,path_val,train_perc,val_perc,shuffle)

        # Get number of images
        self.__num_train_images = len(self.__train_xs)
        self.__num_val_images = len(self.__val_xs)
        print("Number training images: %d" % self.__num_train_images)
        print("Number validation images: %d" % self.__num_val_images)

    def shuffleData(self):
        '''Shuffle all data in memory'''
        # Shuffle data
        c = list(zip(self.__xs, self.__ys))
        random.shuffle(c)
        self.__xs, self.__ys = zip(*c)

        if self.__split_training == True:
            # Training set 80%
            self.__train_xs = self.__xs[:int(len(self.__xs) * self.__train_perc)]
            self.__train_ys = self.__ys[:int(len(self.__xs) * self.__train_perc)]

            # Validation set 20%
            self.__val_xs = self.__xs[-int(len(self.__xs) * self.__val_perc):]
            self.__val_ys = self.__ys[-int(len(self.__xs) * self.__val_perc):]
        else:
            # Training set 100%
            self.__train_xs = self.__xs
            self.__train_ys = self.__ys

    def LoadTrainBatch(self, batch_size, crop_start=126, crop_end=226, should_augment=False):
        '''Load training batch, if batch_size=-1 load all dataset'''
        x_out = []
        y_out = []

        # If batch_size is -1 load the whole thing
        if batch_size == -1:
            batch_size = self.__num_train_images

        # Populate batch
        for i in range(0, batch_size):
            # Load image
            # image = scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images], mode="RGB")
            image = self.__train_xs[(self.__train_batch_pointer + i) % self.__num_train_images]
            # Crop top, resize to 66x200 and divide by 255.0
            image = scipy.misc.imresize(image[crop_start:crop_end], [66, 200]) / 255.0
            x_out.append(image)
            y_out.append([self.__train_ys[(self.__train_batch_pointer + i) % self.__num_train_images]])
            self.__train_batch_pointer += batch_size

        # Augment dataset if needed
        if should_augment == True:
            # Augment training batch
            augmented_batch = self.__augment.augment(list(zip(x_out,y_out)))
            # Expand zip into list
            x_out, y_out = map(list, zip(*augmented_batch))

        return x_out, y_out

    def LoadValBatch(self, batch_size, crop_start=126, crop_end=226):
        '''Load validation batch, if batch_size=-1 load all dataset'''
        x_out = []
        y_out = []

        # If batch_size is -1 load the whole thing
        if batch_size == -1:
            batch_size = self.__num_val_images

        for i in range(0, batch_size):
            # Load image
            # image = scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images], mode="RGB")
            image = self.__val_xs[(self.__val_batch_pointer + i) % self.__num_val_images]
            # Crop top, resize to 66x200 and divide by 255.0
            image = scipy.misc.imresize(image[crop_start:crop_end], [66, 200]) / 255.0
            x_out.append(image)
            y_out.append([self.__val_ys[(self.__val_batch_pointer + i) % self.__num_val_images]])
            self.__val_batch_pointer += batch_size
        return x_out, y_out

    def get_num_images(self):
        return self.__num_images

    @staticmethod
    def save_tfrecord(list_tups_train, filename='newTFRecord.tfrec'):
        '''Save batch to a TFRecord file'''
        writer = tf.python_io.TFRecordWriter(filename)
        # Iterate on batch
        for (tup_element) in list_tups_train:
            img, steer = tup_element
            # Convert to numpy array to get bytes more easy
            shape_info = np.array(img.shape)

            # Describe record entry
            example = tf.train.Example(
                # Record description
                features=tf.train.Features(
                    feature={
                        # A Feature contains one of either a int64_list, float_list, or bytes_list
                        'label': tf.train.Feature(
                            float_list=tf.train.FloatList(value=[steer])),
                        'shape': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[shape_info.tobytes()])),
                        'image': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[img.tobytes()])),
                    }))

            # use the proto object to serialize the example to a string
            serialized = example.SerializeToString()
            # write the serialized object to disk
            writer.write(serialized)

        # Close file
        writer.close()

    @staticmethod
    def save_hdf5(list_tups_train, filename='newHdf5.h5'):
        '''Save batch to a hdf5 file'''
        ys = []
        imgs = []
        for (tup_element) in list_tups_train:
            img, steer = tup_element
            ys.append(steer)
            imgs.append(img)

        # Convert to numpy arrays
        list_labels_ndarray = np.asarray(ys)
        list_imgs_ndarray = np.asarray(imgs)
        # Should have shape 1111,
        list_labels_ndarray = list_labels_ndarray.reshape(list_labels_ndarray.size)

        file = h5py.File(filename, 'w')  # 'a' is append
        # Create Training group
        group = file.create_group("Train")
        # Dataset must be resizable and chunked
        dataset_label = group.create_dataset("Labels", list_labels_ndarray.shape, maxshape=(None,), chunks=True)
        dataset_imgs = group.create_dataset("Images", list_imgs_ndarray.shape, maxshape=(None, 256, 256, 3),
                                            chunks=True)

        # Copy data
        dataset_label[...] = list_labels_ndarray
        dataset_imgs[...] = list_imgs_ndarray

        # Close and save data
        file.flush()
        file.close()


    def get_dataset_set(self):
        '''Get all training+validation set'''
        return list(self.__xs), list(self.__ys)


    def handle_file_dataset(self, path_train, path_val='', train_perc=0.8, val_perc=0.2, shuffle=True):
        '''Handle loading HDF5 and LMDB files'''

        # Check if validation-set exist if not partition from training
        has_validation = not path_val == ''
        if ".h5" in path_train:
            print('HDF5 file')
            # Read hdf5
            self.__file = h5py.File(path_train, 'a')
            # Check if the dataset exist
            exist_train = "/Train/Labels" in self.__file
            if exist_train:
                # Initialize pre-existing datasets
                self.__dataset_imgs = self.__file["/Train/Images"]
                self.__dataset_label = self.__file["/Train/Labels"]

                self.__xs = list(self.__dataset_imgs)
                self.__ys = list(self.__dataset_label)

                self.__num_images = len(self.__xs)

                # Create a zip list with images and angles
                c = list(zip(self.__xs, self.__ys))

                # Shuffle data
                if shuffle:
                    random.shuffle(c)

                # Split the items on c
                self.__xs, self.__ys = zip(*c)

                # Check if validation set is not given
                if not has_validation:
                    print('Spliting training and validation')
                    self.__split_training = True
                    # Training set 80%
                    self.__train_xs = self.__xs[:int(len(self.__xs) * train_perc)]
                    self.__train_ys = self.__ys[:int(len(self.__xs) * train_perc)]

                    # Validation set 20%
                    self.__val_xs = self.__xs[-int(len(self.__xs) * val_perc):]
                    self.__val_ys = self.__ys[-int(len(self.__xs) * val_perc):]
                else:
                    print('Load validation dataset')
                    self.__split_training = False
                    # Read hdf5
                    self.__file_val = h5py.File(path_val, 'a')
                    # Check if the dataset exist
                    exist_val = "/Train/Labels" in self.__file_val
                    if exist_val:
                        # Training set 100%
                        self.__train_xs = self.__xs
                        self.__train_ys = self.__ys

                        # Initialize pre-existing datasets
                        self.__dataset_imgs_val = self.__file_val["/Train/Images"]
                        self.__dataset_label_val = self.__file_val["/Train/Labels"]

                        self.__val_xs = list(self.__dataset_imgs_val)
                        self.__val_ys = list(self.__dataset_label_val)
            else:
                raise RuntimeError('Train dataset not found on hdf5')
        else:
            print('LMDB file')
            env = lmdb.open(path_train, readonly=True)

            # Iterate file and load items on memory
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    key_str = key.decode('ascii')
                    if 'label' in key_str:
                        self.__ys.append(np.float32(np.asscalar(np.frombuffer(value, dtype=np.float32, count=1))))
                    else:
                        # Get shape information from key name
                        info_key = key_str.split('_')
                        # Get image shape [2:None] means from index 2 to the end
                        shape_img = tuple(map(lambda x: int(x), info_key[2:None]))
                        self.__xs.append(np.frombuffer(value, dtype=np.uint8).reshape(shape_img).astype(np.float32))

            self.__num_images = len(self.__xs)

            # Create a zip list with images and angles
            c = list(zip(self.__xs, self.__ys))

            # Shuffle data
            if shuffle:
                random.shuffle(c)

            # Split the items on c
            self.__xs, self.__ys = zip(*c)

            # Check if validation set is not given
            if not has_validation:
                print('Spliting training and validation')
                self.__split_training = True
                # Training set 80%
                self.__train_xs = self.__xs[:int(len(self.__xs) * train_perc)]
                self.__train_ys = self.__ys[:int(len(self.__xs) * train_perc)]

                # Validation set 20%
                self.__val_xs = self.__xs[-int(len(self.__xs) * val_perc):]
                self.__val_ys = self.__ys[-int(len(self.__xs) * val_perc):]
            else:
                print('Load validation dataset')
                self.__split_training = False
                # Read lmdb
                env = lmdb.open(path_val, readonly=True)

                # Training set 100%
                self.__train_xs = self.__xs
                self.__train_ys = self.__ys

                # Iterate file and load items on memory
                with env.begin() as txn:
                    cursor = txn.cursor()
                    for key, value in cursor:
                        key_str = key.decode('ascii')
                        if 'label' in key_str:
                            self.__val_ys.append(np.float32(np.asscalar(np.frombuffer(value, dtype=np.float32, count=1))))
                        else:
                            # Get shape information from key name
                            info_key = key_str.split('_')
                            # Get image shape [2:None] means from index 2 to the end
                            shape_img = tuple(map(lambda x: int(x), info_key[2:None]))
                            self.__val_xs.append(np.frombuffer(value, dtype=np.uint8).reshape(shape_img).astype(np.float32))


    @staticmethod
    def get_list_from_file(file_list):
        ''''Get all lines from a text file'''
        list_files = []
        with open(file_list) as f:
            for line in f:
                list_files.append(line.strip())
        return list_files
