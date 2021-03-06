from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import re
import input_image

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 50,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '',
                           """Path to the data directory.""")

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 1.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.00001  # Initial learning rate.

# Global constants describing the data set.
IMAGE_HEIGHT = input_image.IMAGE_HEIGHT
IMAGE_WIDTH = input_image.IMAGE_WIDTH
NUM_CLASSES = input_image.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = input_image.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = input_image.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'Hike'


def activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))


def variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
    
    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var


def variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    
    Returns:
        Variable Tensor
    """
    var = variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inputs(eval_data):
    """Construct input for the specified purpose.
    
    Args:
        eval_data: bool, indicating whether to load data for evaluation or training.
        data_dir: string, directory where binary data of images is stored.
    
    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS] size.
        labels: Labels. 1D tensor of [batch_size] size.
    
    Raises:
        IOError : If image file cannot be loaded
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    images, labels = input_image.data_input(eval_data=eval_data, data_dir=FLAGS.data_dir, batchsize=FLAGS.batch_size)

    return images, labels


def inference(images):
    """Build the CNN Model.

    Args:
        images: Images returned from input().
    
    Returns:
        Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = variable_with_weight_decay('weights',
                                            shape=[5, 5, 3, 64],
                                            stddev=5e-2,
                                            wd=0.0)
        conv = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # # norm1
    # norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #                   name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = variable_with_weight_decay('weights',
                                            shape=[5, 5, 64, 128],
                                            stddev=5e-2,
                                            wd=0.0)
        conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        activation_summary(conv2)

    # # norm2
    # norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #                   name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 128, 256],
                                            stddev=5e-2,
                                            wd=0.0)
        conv = tf.nn.conv2d(pool2, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
        activation_summary(conv3)

    # pool3
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    # local4
    with tf.variable_scope('local4') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool3, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = variable_with_weight_decay('weights', shape=[dim, 1024],
                                             stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        activation_summary(local4)

    # local5
    with tf.variable_scope('local5') as scope:
        weights = variable_with_weight_decay('weights', shape=[1024, 512],
                                             stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        local5 = tf.nn.relu(tf.matmul(local4, weights) + biases, name=scope.name)
        activation_summary(local5)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        weights = variable_with_weight_decay('weights', [512, NUM_CLASSES],
                                             stddev=1 / 512.0, wd=0.0)
        biases = variable_on_cpu('biases', [NUM_CLASSES],
                                 tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local5, weights), biases, name=scope.name)
        activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.
    
    Add summary for "Loss" and "Loss/avg".
    Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]

    Returns:
        Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def add_loss_summaries(total_loss):
    """Add summaries for losses in the model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + '_raw_', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    """Train the model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
                    processed.
    Returns:
        train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
