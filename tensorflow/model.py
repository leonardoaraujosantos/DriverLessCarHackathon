import tensorflow as tf
import model_util as util


class DrivingModel(object):
    # We're defining the model(ops) structure inside the __init__ because we want to avoid having multiple instances
    # of the same ops by mistake.
    def __init__(self, input_=None, use_placeholder=True, training_mode=False):
        self.__x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3], name='IMAGE_IN')
        self.__y_ = tf.placeholder(tf.float32, shape=[None, 1], name='LABEL_IN')
        self.__dropout_prob = tf.placeholder(tf.float32, name='drop_prob')
        self.__use_placeholder = use_placeholder
        self.__trainmode = tf.placeholder(tf.bool, name='train_mode')

        """ START CHANGING MODEL HERE """
        # CONV 1
        if use_placeholder:
            self.__conv1 = tf.layers.conv2d(self.__x, filters=24, kernel_size=5, strides=2, name='conv1')
        else:
            self.__conv1 = tf.layers.conv2d(input_, filters=24, kernel_size=5, strides=2, padding='valid', name='conv1')
        self.__conv1_bn = tf.layers.batch_normalization(self.__conv1, training=self.__trainmode, name='bn_c1')
        self.__conv1_act = tf.nn.relu(self.__conv1_bn)

        # CONV 2
        self.__conv2 = tf.layers.conv2d(self.__conv1_act, filters=36, kernel_size=5, strides=2, padding='valid', name='conv2')
        self.__conv2_bn = tf.layers.batch_normalization(self.__conv2, training=self.__trainmode, name='bn_c2')
        self.__conv2_act = tf.nn.relu(self.__conv2_bn)

        # CONV 3
        self.__conv3 = tf.layers.conv2d(self.__conv2_act, filters=48, kernel_size=5, strides=2, padding='valid', name='conv3')
        self.__conv3_bn = tf.layers.batch_normalization(self.__conv3, training=self.__trainmode, name='bn_c3')
        self.__conv3_act = tf.nn.relu(self.__conv3_bn)

        # CONV 4
        self.__conv4 = tf.layers.conv2d(self.__conv3_act, filters=64, kernel_size=3, strides=1, padding='valid', name='conv4')
        self.__conv4_bn = tf.layers.batch_normalization(self.__conv4, training=self.__trainmode, name='bn_c4')
        self.__conv4_act = tf.nn.relu(self.__conv4_bn)

        # CONV 5
        self.__conv5 = tf.layers.conv2d(self.__conv4_act, filters=64, kernel_size=3, strides=1, padding='valid', name='conv5')
        self.__conv5_bn = tf.layers.batch_normalization(self.__conv5, training=self.__trainmode, name='bn_c5')
        self.__conv5_act = tf.nn.relu(self.__conv5_bn)

        self.__conv5_flat = tf.layers.flatten(self.__conv5_act, name='flatten_conv5')

        # Fully Connect 1 with dropout
        self.__fc1 = tf.layers.dense(self.__conv5_flat, units=1164, activation=tf.nn.relu, name='fc1')
        self.__fc1_drop = tf.layers.dropout(self.__fc1, rate=self.__dropout_prob, training=self.__trainmode)

        # Fully Connect 2 with dropout
        self.__fc2 = tf.layers.dense(self.__fc1_drop, units=100, activation=tf.nn.relu, name='fc2')
        self.__fc2_drop = tf.layers.dropout(self.__fc2, rate=self.__dropout_prob, training=self.__trainmode)

        # Fully Connect 3 with dropout
        self.__fc3 = tf.layers.dense(self.__fc2_drop, units=50, activation=tf.nn.relu, name='fc3')
        self.__fc3_drop = tf.layers.dropout(self.__fc3, rate=self.__dropout_prob, training=self.__trainmode)

        # Fully Connect 4 with dropout
        self.__fc4 = tf.layers.dense(self.__fc3_drop, units=10, activation=tf.nn.relu, name='fc4')
        self.__fc4_drop = tf.layers.dropout(self.__fc4, rate=self.__dropout_prob, training=self.__trainmode)

        # Output
        self.__y = tf.layers.dense(self.__fc4_drop, units=1, name='output_layer')

        """ END OF CHANGING YOUR MODEL """

    @property
    def output(self):
        return self.__y

    @property
    def input(self):
        if self.__use_placeholder:
            return self.__x
        else:
            return None

    @property
    def label_in(self):
        if self.__use_placeholder:
            return self.__y_
        else:
            return None

    @property
    def dropout_control(self):
        return self.__dropout_prob

    @property
    def train_mode(self):
        return self.__trainmode
    
    @property
    def conv5(self):
        return self.__conv5_act

    @property
    def conv4(self):
        return self.__conv4_act

    @property
    def conv3(self):
        return self.__conv3_act

    @property
    def conv2(self):
        return self.__conv2_act

    @property
    def conv1(self):
        return self.__conv1_act


# http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
# http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf
# https://stackoverflow.com/questions/35980044/getting-the-output-shape-of-deconvolution-layer-using-tf-nn-conv2d-transpose-in
# https://arxiv.org/pdf/1411.4038.pdf
class DrivingModelAutoEncoder(object):
    def __init__(self, input_=None, use_placeholder=True, training_mode=False):
        self.__x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3], name='IMAGE_IN')
        self.__y_ = tf.placeholder(tf.float32, shape=[None, 1], name='LABEL_IN')
        self.__dropout_prob = tf.placeholder(tf.float32, name='drop_prob')
        self.__use_placeholder = use_placeholder

        ##### ENCODER
        # Calculating the convolution output:
        # https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html
        # H_out = 1 + (H_in+(2*pad)-K)/S
        # W_out = 1 + (W_in+(2*pad)-K)/S

        # CONV1: Input 66x200x3 after CONV 5x5 P:0 S:2 H_out: 1 + (66-5)/2 = 31, W_out= 1 + (200-5)/2=98
        if use_placeholder:
            self.__conv1 = tf.layers.conv2d(self.__x, filters=24, kernel_size=5, strides=2, name='conv1')
        else:
            self.__conv1 = tf.layers.conv2d(input_, filters=24, kernel_size=5, strides=2, padding='valid', name='conv1')
        self.__conv1_bn = tf.layers.batch_normalization(self.__conv1, training=training_mode, name='bn_c1')
        self.__conv1_act = tf.nn.relu(self.__conv1_bn)

        # CONV2: Input 31x98x24 after CONV 5x5 P:0 S:2 H_out: 1 + (31-5)/2 = 14, W_out= 1 + (200-5)/2=47
        self.__conv2 = tf.layers.conv2d(self.__conv1_act, filters=36, kernel_size=5, strides=2, padding='valid', name='conv2')
        self.__conv2_bn = tf.layers.batch_normalization(self.__conv2, training=training_mode, name='bn_c2')
        self.__conv2_act = tf.nn.relu(self.__conv2_bn)

        # CONV3: Input 14x47x36 after CONV 5x5 P:0 S:2 H_out: 1 + (14-5)/2 = 5, W_out= 1 + (47-5)/2=22
        self.__conv3 = tf.layers.conv2d(self.__conv2_act, filters=48, kernel_size=5, strides=2, padding='valid', name='conv3')
        self.__conv3_bn = tf.layers.batch_normalization(self.__conv3, training=training_mode, name='bn_c3')
        self.__conv3_act = tf.nn.relu(self.__conv3_bn)

        # CONV4: Input 5x22x48 after CONV 3x3 P:0 S:1 H_out: 1 + (5-3)/1 = 3, W_out= 1 + (22-3)/1=20
        self.__conv4 = tf.layers.conv2d(self.__conv3_act, filters=64, kernel_size=3, strides=1, padding='valid', name='conv4')
        self.__conv4_bn = tf.layers.batch_normalization(self.__conv4, training=training_mode, name='bn_c4')
        self.__conv4_act = tf.nn.relu(self.__conv4_bn)

        # CONV5: Input 3x20x64 after CONV 3x3 P:0 S:1 H_out: 1 + (3-3)/1 = 1, W_out= 1 + (20-3)/1=18
        self.__conv5 = tf.layers.conv2d(self.__conv4_act, filters=64, kernel_size=3, strides=1, padding='valid', name='conv5')
        self.__conv5_bn = tf.layers.batch_normalization(self.__conv5, training=training_mode, name='bn_c5')
        self.__conv5_act = tf.nn.relu(self.__conv5_bn)

        ##### DECODER (At this point we have 1x18x64
        # Kernel, output size, in_volume, out_volume, stride
        self.__conv_t5_out = util.conv2d_transpose(self.__conv5_act, (3, 3), (3, 20), 64, 64, 1, name="dconv1", do_summary=False)
        self.__conv_t5_out_bn = tf.layers.batch_normalization(self.__conv_t5_out, training=training_mode, name='bn_t_c5')
        self.__conv_t5_out_act = tf.nn.relu(self.__conv_t5_out_bn)

        self.__conv_t4_out = util.conv2d_transpose(self.__conv_t5_out_act, (3, 3), (5, 22), 64, 48, 1, name="dconv2", do_summary=False)
        self.__conv_t4_out_bn = tf.layers.batch_normalization(self.__conv_t4_out, training=training_mode, name='bn_t_c4')
        self.__conv_t4_out_act = tf.nn.relu(self.__conv_t4_out_bn)

        self.__conv_t3_out = util.conv2d_transpose(self.__conv_t4_out_act, (5, 5), (14, 47), 48, 36, 2, name="dconv3", do_summary=False)
        self.__conv_t3_out_bn = tf.layers.batch_normalization(self.__conv_t3_out, training=training_mode, name='bn_t_c3')
        self.__conv_t3_out_act = tf.nn.relu(self.__conv_t3_out_bn)

        self.__conv_t2_out = util.conv2d_transpose(self.__conv_t3_out_act, (5, 5), (31, 98), 36, 24, 2, name="dconv4", do_summary=False)
        self.__conv_t2_out_bn = tf.layers.batch_normalization(self.__conv_t2_out, training=training_mode, name='bn_t_c2')
        self.__conv_t2_out_act = tf.nn.relu(self.__conv_t2_out_bn)

        self.__conv_t1_out = util.conv2d_transpose(self.__conv_t2_out_act, (5, 5), (66, 200), 24, 3, 2, name="dconv5", do_summary=False)
        self.__conv_t1_out_bn = tf.layers.batch_normalization(self.__conv_t1_out, training=training_mode, name='bn_t_c1')
        self.__y = tf.nn.relu(self.__conv_t1_out_bn)

    @property
    def output(self):
        return self.__y

    @property
    def input(self):
        if self.__use_placeholder:
            return self.__x
        else:
            return None

    @property
    def label_in(self):
        if self.__use_placeholder:
            return self.__y_
        else:
            return None

    @property
    def dropout_control(self):
        return self.__dropout_prob

    @property
    def conv5(self):
        return self.__conv5_act

    @property
    def conv4(self):
        return self.__conv4_act

    @property
    def conv3(self):
        return self.__conv3_act

    @property
    def conv2(self):
        return self.__conv2_act

    @property
    def conv1(self):
        return self.__conv1_act
