from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
NUM_CLASSES = 3

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 15000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 3000

TRAIN_FILENAME = 'train'
EVAL_FILENAME = 'validation'


def getImage(filename):
    """Function to tell TensorFlow how to read a single image from input file

    Args:
        filename: path for the file to load

    Returns:
        image:
    """
    # convert filenames to a queue for an input pipeline.
    filenameQ = tf.train.string_input_producer([filename])

    # object to read records
    recordReader = tf.TFRecordReader()

    # read the full set of features for a single example
    key, full = recordReader.read(filenameQ)

    # parse the full example into its' component features.
    features = tf.parse_single_example(
        full,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/channels': tf.FixedLenFeature([], tf.int64),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/format': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
        })

    # now we are going to manipulate the label and image features

    label = features['image/class/label']
    image_buffer = features['image/encoded']

    # Decode the jpeg
    with tf.name_scope('decode_jpeg', [image_buffer], None):
        # decode
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        image = tf.image.resize_images(image, size=[IMAGE_HEIGHT, IMAGE_WIDTH])

        # and convert to single precision data type
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # re-define label as a "one-hot" vector
    # it will be [0,1] or [1,0] here.
    # This approach can easily be extended to more classes.
    # label = tf.stack(tf.one_hot(label - 1, NUM_CLASSES))
    label = label-1

    return image, label


def generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
                            in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images, max_outputs=10)

    return images, tf.reshape(label_batch, [batch_size])


def data_input(eval_data, data_dir, batchsize):
    if eval_data:
        filename = os.path.join(data_dir, EVAL_FILENAME)
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    else:
        filename = os.path.join(data_dir, TRAIN_FILENAME)
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

    image, label = getImage(filename)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)
    
    return generate_image_and_label_batch(image=image, label=label,min_queue_examples=min_queue_examples,
                                          batch_size=batchsize, shuffle=True)

