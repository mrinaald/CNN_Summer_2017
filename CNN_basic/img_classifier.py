from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

from PIL import Image

from random import shuffle

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_boolean('train', False, 'Flag for training the model')
tf.app.flags.DEFINE_boolean('evaluate', False, 'Flag for evaluating the model')
tf.app.flags.DEFINE_string('model_dir', '/tmp/img_classifier', 'Output directory of the model')
tf.app.flags.DEFINE_string('train_dir', '', 'Train data directory')
tf.app.flags.DEFINE_string('eval_dir', '', 'Evaluation data directory')

FLAGS = tf.app.flags.FLAGS

img_size = 128
channels = 3


def load_images(folder, genre, purpose):
    img_files = os.listdir(os.path.join(folder, genre, purpose))

    dataset = np.ndarray(shape=(len(img_files), img_size, img_size, channels), dtype=np.float32)

    num_images = 0
    for image in img_files:
        image_file = os.path.join(folder, genre, purpose, image)
        try:
            image_data = Image.open(image_file)  # opening image
            image_data = image_data.convert('RGB')      # converting to RGB from RGBA(if exists)
            image_data = image_data.resize((img_size, img_size))  # resizing the image

            dataset[num_images, :, :, :] = np.array(image_data)

            # out = Image.fromarray(np.array(dataset[num_images, :, :, :], dtype=np.uint8))
            # if out.mode != 'RGB' :
            #     out = out.convert('RGB')
            #
            # # this method requires an RGB image, that is why it is converted above
            # out.save('./' + folder + '/save/' + genre + '_' + purpose +'_' + str(num_images) + '.jpg')

            num_images = num_images + 1

        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    return dataset

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 128x128 pixels, and have three color channels
    input_layer = tf.reshape(features, [-1, img_size, img_size, channels])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 128, 128, 3]
    # Output Tensor Shape: [batch_size, 128, 128, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 128, 128, 32]
    # Output Tensor Shape: [batch_size, 64, 64, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 3x3 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 64, 64, 32]
    # Output Tensor Shape: [batch_size, 64, 64, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 64, 64, 64]
    # Output Tensor Shape: [batch_size, 32, 32, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3
    # Computes 128 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 32, 32, 64]
    # Output Tensor Shape: [batch_size, 32, 32, 128]
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #3
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 32, 32, 128]
    # Output Tensor Shape: [batch_size, 16, 16, 128]
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # Convolutional Layer #4
    # Computes 128 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 16, 16, 128]
    # Output Tensor Shape: [batch_size, 16, 16, 256]
    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=256,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #4
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 16, 16, 256]
    # Output Tensor Shape: [batch_size, 8, 8, 256]
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 8, 8, 256]
    # Output Tensor Shape: [batch_size, 8 * 8 * 256]
    pool4_flat = tf.reshape(pool4, [-1, 8 * 8 * 256])

    # Dense Layer 1
    # Densely connected layer with 5632 [≈ (8*8*256)/3 ] neurons (512*11 = 5632)
    # Input Tensor Shape: [batch_size, 8 * 8 * 256]
    # Output Tensor Shape: [batch_size, 5632]
    dense1 = tf.layers.dense(inputs=pool4_flat, units=5632, activation=tf.nn.relu)

    # Dense Layer 2
    # Densely connected layer with 1870 [≈ (5632)/3 ] neurons
    # Input Tensor Shape: [batch_size, 5632]
    # Output Tensor Shape: [batch_size, 1870]
    dense2 = tf.layers.dense(inputs=dense1, units=1870, activation=tf.nn.relu)

    # Add dropout operation; 0.7 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense2, rate=0.3, training=mode == learn.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1870]
    # Output Tensor Shape: [batch_size, 3]
    logits = tf.layers.dense(inputs=dropout, units=3)

    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="SGD")

    # Generate Predictions
    predictions = {
        "classes": tf.argmax(
            input=logits, axis=1),
        "probabilities": tf.nn.softmax(
            logits, name="softmax_tensor")
    }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)

def main(unused_argv):
    # Load training data
    train_data1 = load_images(folder='HikeImages', genre='memes', purpose='train')
    train_labels1 = np.zeros(shape=len(train_data1), dtype=np.int32)
    print('Train Data 1:', len(train_data1), ' Train Labels 1:', len(train_labels1))
    train_data2 = load_images(folder='HikeImages', genre='quotes', purpose='train')
    train_labels2 = np.ones(shape=(len(train_data2)), dtype=np.int32)
    print('Train Data 2:', len(train_data2), ' Train Labels 2:', len(train_labels2))
    train_data3 = load_images(folder='HikeImages', genre='others', purpose='train')
    train_labels3 = 2 * np.ones(shape=(len(train_data3)), dtype=np.int32)
    print('Train Data 3:', len(train_data3), ' Train Labels 3:', len(train_labels3))


    train_data_temp = np.concatenate((train_data1, train_data2,train_data3))
    train_labels_temp = np.concatenate((train_labels1, train_labels2,train_labels3))

    train_data = np.ndarray(shape=(len(train_data_temp), img_size, img_size, channels), dtype=np.float32)
    train_labels = np.ndarray(shape=(len(train_data_temp)), dtype=np.int32)

    index_shuf = list(range(len(train_data_temp)))
    shuffle(index_shuf)
    for i in index_shuf:
        train_data[i, :, :, :] = train_data_temp[i, :, :, :]
        train_labels[i] = train_labels_temp[i]

    del train_data1
    del train_data2
    del train_data3
    del train_labels1
    del train_labels2
    del train_labels3
    del train_data_temp
    del train_labels_temp

    print('Train Data:', len(train_data))
    print('Train Labels:', len(train_labels))
    print('Train Data Done')

    # Load evaluation data
    # eval_data1 = load_images(folder='HikeImages', genre='memes', purpose='test')
    # eval_labels1 = np.zeros(shape=len(eval_data1), dtype=np.int32)
    # print('Eval Data 1:', len(eval_data1), ' Eval Labels 1:', len(eval_labels1))
    # eval_data2 = load_images(folder='HikeImages', genre='quotes', purpose='test')
    # eval_labels2 = np.ones(shape=(len(eval_data2)), dtype=np.int32)
    # print('Eval Data 2:', len(eval_data2), ' Eval Labels 2:', len(eval_labels2))
    # eval_data3 = load_images(folder='HikeImages', genre='others', purpose='test')
    # eval_labels3 = 2 * np.ones(shape=(len(eval_data3)), dtype=np.int32)
    # print('Eval Data 3:', len(eval_data3), ' Eval Labels 3:', len(eval_labels3))

    # eval_data_temp = np.concatenate((eval_data1, eval_data2, eval_data3))
    # eval_labels_temp = np.concatenate((eval_labels1, eval_labels2, eval_labels3))

    # eval_data = np.ndarray(shape=(len(eval_data_temp), img_size, img_size), dtype=np.float32)
    # eval_labels = np.ndarray(shape=(len(eval_data_temp)), dtype=np.int32)

    # index_shuf = list(range(len(eval_data_temp)))
    # shuffle(index_shuf)
    # for i in index_shuf:
    #     eval_data[i, :, :] = eval_data_temp[i, :, :]
    #     eval_labels[i] = eval_labels_temp[i]

    # print('Eval Data:', len(eval_data_temp))
    # print('Eval Labels:', len(eval_labels_temp))
    # print('Eval Data Done')

    # Load prediction data
    # pred_data = load_images(folder='HikeImages', genre='prediction', purpose='')
    # print('Prediction Data Done')

    # Create the Estimator
    mnist_classifier = learn.Estimator(
        model_fn=cnn_model_fn, model_dir="./rgb_classifier_3")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    mnist_classifier.fit(
        x=train_data,
        y=train_labels,
        batch_size=100,
        steps=1500,
        monitors=[logging_hook])

    # Configure the accuracy metric for evaluation
    # metrics = {
    #     "accuracy":
    #         learn.MetricSpec(
    #             metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    # }

    metrics = {
        "false positives": learn.MetricSpec(metric_fn=tf.metrics.false_positives, prediction_key="classes"),
        "false negatives": learn.MetricSpec(metric_fn=tf.metrics.false_negatives, prediction_key="classes"),
        "true positives": learn.MetricSpec(metric_fn=tf.metrics.true_positives, prediction_key="classes"),
        "accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }

    # Evaluate the model and print results
    # eval_results = mnist_classifier.evaluate(
    #     x=eval_data, y=eval_labels, metrics=metrics)
    # print(eval_results)

    # prediction = list(mnist_classifier.predict(x=pred_data))
    # prediction = mnist_classifier.pred
    # print(prediction)

if __name__ == "__main__":
    tf.app.run()
