# https://www.tensorflow.org/programmers_guide/variables
import argparse
import os
import tensorflow as tf
from driving_data import HandleData
import model
import model_util as util

# Example: python train_unsupervised_tfrecord.py --batch=100 --learning_rate=0.001
parser = argparse.ArgumentParser(description='Train network')
parser.add_argument('--input', type=str, required=False, default='DrivingData.h5', help='Training hdf5/lmdb file')
parser.add_argument('--input_val', type=str, required=False, default='', help='Validation hdf5 file')
parser.add_argument('--gpu', type=int, required=False, default=0, help='GPU number (-1) for CPU')
parser.add_argument('--checkpoint_dir', type=str, required=False, default='', help='Load checkpoint')
parser.add_argument('--logdir', type=str, required=False, default='./logs', help='Tensorboard log directory')
parser.add_argument('--savedir', type=str, required=False, default='./save', help='Tensorboard checkpoint directory')
parser.add_argument('--epochs', type=int, required=False, default=600, help='Number of epochs')
parser.add_argument('--batch', type=int, required=False, default=400, help='Batch size')
parser.add_argument('--learning_rate', type=float, required=False, default=0.0001, help='Initial learning rate')
args = parser.parse_args()


def train_network(input_train_hdf5, input_val_hdf5, gpu, pre_trained_checkpoint, epochs, batch_size, logs_path, save_dir):

    # Create log directory if it does not exist
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    # Set enviroment variable to set the GPU to use
    if gpu != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        print('Set tensorflow on CPU')
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Define number of epochs and batch size, where to save logs, etc...
    iter_disp = 10
    start_lr = args.learning_rate

    # Avoid allocating the whole memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    # Regularization value
    L2NormConst = 0.001

    # Build model and get references to placeholders
    driving_model = model.DrivingModelAutoEncoder(training_mode=True)
    model_in = driving_model.input
    model_out = driving_model.output
    model_drop = driving_model.dropout_control

    # Add input image/steering angle on summary
    tf.summary.image("input_image", model_in, 10)
    tf.summary.image("output_image", model_out, 10)

    # Loss is binary cross-entropy
    # Get all model "parameters" that are trainable
    train_vars = tf.trainable_variables()
    # Trying now L2 loss (maybe not good idea)
    with tf.name_scope("L2_LOSS"):
        #cross_entropy = -1. * model_in * tf.log(model_out) - (1. - model_in) * tf.log(1. - model_out)
        #loss = tf.reduce_mean(cross_entropy)
        #loss = tf.nn.l2_loss(model_in-model_out)
        loss = tf.reduce_mean(util.huber_loss(model_out, model_in))

    # Solver configuration
    # Get ops to update moving_mean and moving_variance from batch_norm
    # Reference: https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.name_scope("Solver"):
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = start_lr
        # decay every 10000 steps with a base of 0.96
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   5000, 0.9, staircase=True)

        # Basically update the batch_norm moving averages before the training step
        # http://ruishu.io/2016/12/27/batchnorm/
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Initialize all random variables (Weights/Bias)
    sess.run(tf.global_variables_initializer())

    # Load checkpoint if needed
    if pre_trained_checkpoint:
        # Load tensorflow model
        print("Loading pre-trained model: %s" % args.checkpoint_dir)
        # Create saver object to save/load training checkpoint
        saver = tf.train.Saver(max_to_keep=None)
        saver.restore(sess, args.checkpoint_dir)
    else:
        # Just create saver for saving checkpoints
        saver = tf.train.Saver(max_to_keep=None)

    # Monitor loss, learning_rate, global_step, etc...
    tf.summary.scalar("loss_train", loss)
    tf.summary.scalar("learning_rate", learning_rate)
    tf.summary.scalar("global_step", global_step)
    # merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    # Configure where to save the logs for tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    data = HandleData(path=input_train_hdf5, path_val=input_val_hdf5)
    num_images_epoch = int(data.get_num_images() / batch_size)
    print('Num samples',data.get_num_images(), 'Iterations per epoch:', num_images_epoch, 'batch size:', batch_size)

    # For each epoch
    for epoch in range(epochs):
        for i in range(int(data.get_num_images() / batch_size)):
            # Get training batch
            xs_train, ys_train = data.LoadTrainBatch(batch_size, should_augment=False)

            # Send training batch to tensorflow graph (Dropout enabled)
            train_step.run(feed_dict={model_in: xs_train, model_drop: 0.8})

            # write logs at every iteration
            summary = merged_summary_op.eval(feed_dict={model_in: xs_train, model_drop: 1.0})
            summary_writer.add_summary(summary, epoch * batch_size + i)

        # Save checkpoint after each epoch
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        checkpoint_path = os.path.join(save_dir, "model")
        filename = saver.save(sess, checkpoint_path, global_step=epoch)
        print("Model saved in file: %s" % filename)

        # Shuffle data at each epoch end
        print("Shuffle data")
        data.shuffleData()

    print("Run the command line:\n" \
          "--> tensorboard --logdir=./logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")

if __name__ == "__main__":
    # Call function that implement the auto-pilot
    train_network(args.input, args.input_val, args.gpu,
                  args.checkpoint_dir, args.epochs, args.batch, args.logdir, args.savedir)
