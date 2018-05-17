import argparse
import os
import tensorflow as tf
import time
from driving_data import HandleData
import model
import model_util as util

# Example: python train.py --input=DrivingData.h5 --gpu=0 --checkpoint_dir=save/model.ckpt
parser = argparse.ArgumentParser(description='Train network')
parser.add_argument('--input_list', type=str, required=False, default='FileList.txt', help='Training list of TFRecord files')
parser.add_argument('--input_val', type=str, required=False, default='', help='Validation TFRecord file')
parser.add_argument('--gpu', type=int, required=False, default=0, help='GPU number (-1) for CPU')
parser.add_argument('--checkpoint_dir', type=str, required=False, default='', help='Load checkpoint')
parser.add_argument('--logdir', type=str, required=False, default='./logs', help='Tensorboard log directory')
parser.add_argument('--savedir', type=str, required=False, default='./save', help='Tensorboard checkpoint directory')
parser.add_argument('--epochs', type=int, required=False, default=600, help='Number of epochs')
parser.add_argument('--batch', type=int, required=False, default=400, help='Batch size')
parser.add_argument('--gpu_frac', type=float, required=False, default=0.5, help='Gpu memory fraction')
parser.add_argument('--learning_rate', type=float, required=False, default=0.0001, help='Initial learning rate')
args = parser.parse_args()

def train_network(input_list, input_val_hdf5, gpu, pre_trained_checkpoint, epochs, batch_size, logs_path, save_dir, gpu_frac):

    # Create log directory if it does not exist
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    # Set enviroment variable to set the GPU to use
    if gpu != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        print('Set tensorflow on CPU')
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Get file list
    list_tfrecord_files = HandleData.get_list_from_file(input_list)

    # Create the graph input part (Responsible to load files, do augmentations, etc...)
    images, labels = util.create_input_graph(list_tfrecord_files, epochs, batch_size, do_augment=True)

    # Build Graph
    driving_model = model.DrivingModel(input=images, use_placeholder = False)
    model_out = driving_model.output
    dropout_prob = driving_model.dropout_control

    # Create histogram for labels
    tf.summary.histogram("steer_angle", labels)
    # Add input image/steering angle on summary
    tf.summary.image("input_image", images, 10)

    # Define number of epochs and batch size, where to save logs, etc...
    iter_disp = 10
    start_lr = args.learning_rate

    # Avoid allocating the whole memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    # Regularization value
    L2NormConst = 0.001

    # Loss is mean squared error plus l2 regularization
    # model.y (Model output), model.y_(Labels)
    # tf.nn.l2_loss: Computes half the L2 norm of a tensor without the sqrt
    # output = sum(t ** 2) / 2
    # Get all model "parameters" that are trainable
    train_vars = tf.trainable_variables()
    with tf.name_scope("MSE_Loss_L2Reg"):
        labels = tf.reshape(labels, model_out.shape)
        loss = tf.reduce_mean(tf.square(tf.subtract(labels, model_out))) + tf.add_n(
            [tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
    #with tf.name_scope("Loss_Huber"):
    #    loss = tf.reduce_mean(util.huber_loss(labels, model_out))


    # Get ops to update moving_mean and moving_variance from batch_norm
    # Reference: https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Solver configuration
    with tf.name_scope("Solver"):
        #global_step = tf.Variable(0, name='global_step', trainable=False)
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = start_lr
        # decay every 10000 steps with a base of 0.96
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   10000, 0.5, staircase=True)

        # Basically update the batch_norm moving averages before the training step
        # http://ruishu.io/2016/12/27/batchnorm/
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Initialize all random variables (Weights/Bias)
    # Notice that now we also mention the local_variables, which are used on the queue coordinators
    # And our counters, tensorflow automatically keep a different container of all global variables and local
    # variables
    # https://stackoverflow.com/questions/38910198/what-is-a-local-variable-in-tensorflow
    # tf.GraphKeys.LOCAL_VARIABLES
    # tf.GraphKeys.VARIABLES
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init_op)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Load checkpoint if needed
    if pre_trained_checkpoint:
        # Load tensorflow model
        print("Loading pre-trained model: %s" % args.checkpoint_dir)
        # Create saver object to save/load training checkpoint
        saver = tf.train.Saver(max_to_keep=None)
        saver.restore(sess, args.checkpoint_dir)

        # If learning_rate is set reset the global variable
        assign_globabl_step_op = tf.assign(global_step, 0)
        sess.run(assign_globabl_step_op)

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

    try:
        step = 0
        count_model = 0
        while not coord.should_stop():
            start_time = time.time()

            # Run one step of the model.  The return values are
            # the activations from the `train_op` (which is
            # discarded) and the `loss` op.  To inspect the values
            # of your ops or variables, you may include them in
            # the list passed to sess.run() and the value tensors
            # will be returned in the tuple from the call.
            _, loss_value = sess.run([train_step, loss], feed_dict={dropout_prob: 0.8})

            duration = time.time() - start_time

            # Print an overview fairly often.
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,duration))

            if step % iter_disp == 0:
                # write logs
                summary = merged_summary_op.eval(feed_dict={dropout_prob: 1.0})
                summary_writer.add_summary(summary, batch_size + step)

            # Save model
            if step % 400 == 0:
                # Save checkpoint after each epoch
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                checkpoint_path = os.path.join(save_dir, "model")
                filename = saver.save(sess, checkpoint_path, global_step=count_model)
                print("Model saved in file: %s" % filename)
                count_model += 1

            step += 1

    except tf.errors.OutOfRangeError:
        #print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
        print('error')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

        # Wait for threads to finish.
    coord.join(threads)
    sess.close()



if __name__ == "__main__":
    # Call function that implement the auto-pilot
    train_network(args.input_list, args.input_val, args.gpu,
                  args.checkpoint_dir, args.epochs, args.batch, args.logdir, args.savedir, args.gpu_frac)
