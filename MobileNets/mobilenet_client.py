from __future__ import print_function

import os
import timeit
import pickle

# This is a placeholder for a Google-internal import.
from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

# Flags which can be passed in the command line
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
tf.app.flags.DEFINE_string('input_dir', '',
                           'A directory containing .jpg or .jpeg files to serialize into a '
                           'request json')
tf.app.flags.DEFINE_string('input_image', '', 'A .jpg image file to be predicted')
tf.app.flags.DEFINE_string('model_info_file', '', 'The .pickle dump file in which info for the '
                                                  'MobileNet model is stored')
tf.app.flags.DEFINE_string('label_file', '', 'The file which contains the prediction labels')
tf.app.flags.DEFINE_string('model_name', '', 'The name of the model deployed through tensorflow serving')
FLAGS = tf.app.flags.FLAGS


def load_labels(label_file):
    """Creates the list of possible categories predicted by the model

    Args:
        label_file: The name of file containing the labels.

    Returns:
        List of categories
    """
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def do_inference(hostport, request_dict, model_name, model_info, labels):
    """Tests PredictionService with concurrent requests.

    Args:
        hostport: Host:port address of the PredictionService.
        request_dict: Dictionary for the values to be sent as input to the server.
        model_name: Name of the deployed tensorflow server.
        model_info: Dictionary storing the information related to the trained model.
        labels: List of categories to be used for specifying the label of the prediction.

    Returns:
        The classification error rate.

    Raises:
        IOError: An error occurred processing test data set.
    """
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = 'serving_default'

    request.inputs['key'].CopyFrom(
        tf.contrib.util.make_tensor_proto(request_dict['key'], shape=[1]))
    image_shape = [1, model_info['input_height'], model_info['input_width'], model_info['input_depth']]
    request.inputs['image_bytes'].CopyFrom(
        tf.contrib.util.make_tensor_proto(request_dict['image_bytes'], shape=image_shape))
    start_time = timeit.default_timer()
    res = stub.Predict(request, 27.0)  # wait 27 seconds for response
    elapsed = timeit.default_timer() - start_time
    print('Request_Response Time : ', elapsed)

    key = res.outputs['key'].string_val
    scores = res.outputs['scores'].float_val
    prediction = res.outputs['prediction'].int64_val
    print('\nKey : ', key)
    print('Prediction Value : ', prediction)
    for i in prediction:
        print('Prediction Label : ', labels[i])
    print('Scores : ', scores)


def read_tensor_from_image_file(file_name, input_height, input_width, input_mean, input_std):
    """Tests PredictionService with concurrent requests.

    Args:
        file_name: The name of the image file.
        input_height: The input height required by the trained model.
        input_width:  The input width required by the trained model.
        input_mean: The input mean of the image required by the trained model (for feature scaling).
        input_std: The input standard deviation required by the trained model (for feature scaling).

    Returns:
        The classification error rate.

    Raises:
        IOError: An error occurred processing test data set.
    """
    input_name = "file_reader"
    file_reader = tf.read_file(file_name, input_name)

    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3,
                                           name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3,
                                            name='jpeg_reader')

    image_reader_float = tf.cast(image_reader, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(image_reader_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                             resize_shape_as_int)
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    sess = tf.Session()
    result = sess.run(mul_image)

    return result


def inference_from_file(server, input_image, model_name, model_info, labels):
    """Makes the request dictionary from the input_image

    Args:
        server: The PredictionService host:port
        input_image: The name of the image.
        model_name: The name of the model deployed on the server.
        model_info: The dictionary of information related to the deployed model.
        labels: List of categories to be used for specifying the label of the prediction.
    """
    if not os.path.isfile(input_image):
        print('The input_image ' + input_image + ' does not exist.')
        print('Aborting!!!')
        exit(1)

    try:
        start_time = timeit.default_timer()
        decoded_image = read_tensor_from_image_file(input_image, model_info['input_height'],
                                                    model_info['input_width'], model_info['input_mean'],
                                                    model_info['input_std'])
        elapsed = timeit.default_timer() - start_time
        print('Conversion Time : ', elapsed)

        request_dict = {'key': input_image,
                        'image_bytes': decoded_image
                        }
        do_inference(server, request_dict, model_name, model_info, labels)
    except IOError as e:
        print('Could not read:', input_image, ':', e, '- it\'s ok, skipping.')


def inference_from_dir(server, input_dir, model_name, model_info, labels):
    """Makes the request dictionary from all the images in the input_dir

    Args:
        server: The PredictionService host:port
        input_dir: The directory which contains the images.
        model_name: The name of the model deployed on the server.
        model_info: The dictionary of information related to the deployed model.
        labels: List of categories to be used for specifying the label of the prediction.
    """
    if not os.path.isdir(input_dir):
        print('The input_dir ' + input_dir + ' does not exist.')
        print('Aborting!!!')
        exit(1)

    for in_dir in [input_dir]:
        input_images = os.listdir(in_dir)

        for image in input_images:
            image_handle = os.path.join(in_dir, image)
            try:
                start_time = timeit.default_timer()
                decoded_image = read_tensor_from_image_file(image_handle, model_info['input_height'],
                                                            model_info['input_width'], model_info['input_mean'],
                                                            model_info['input_std'])
                elapsed = timeit.default_timer() - start_time
                print('\nConversion Time : ', elapsed)

                request_dict = {'key': image_handle,
                                'image_bytes': decoded_image
                                }
                do_inference(server, request_dict, model_name, model_info, labels)
            except IOError as e:
                print('Could not read:', image_handle, ':', e, '- it\'s ok, skipping.')


def main(_):
    if not FLAGS.server:
        print('Please specify server using --server=host:port')
        print('Aborting!!!')
        return
    if not FLAGS.model_name:
        print('Please specify model name using --model_name')
        print('Aborting!!!')
        return
    if not FLAGS.input_dir and not FLAGS.input_image:
        print('Please specify either the input_dir or input_image')
        print('Aborting!!!')
        return
    if FLAGS.input_dir and FLAGS.input_image:
        print('Please specify either one of input_dir or input_image')
        print('Aborting!!!')
        return
    if not FLAGS.model_info_file:
        print('Please specify the model info pickle dump file')
        print('Aborting!!!')
        return
    if not FLAGS.label_file:
        print('Please specify the labels file')
        print('Aborting!!!')
        return

    if not os.path.isfile(FLAGS.model_info_file):
        print('File : ' + FLAGS.model_info_file + ' does not exist.')
        print('Aborting!!!')
        return

    # Loading the information related to the deployed model from a 'pickle' dump file
    with open(FLAGS.model_info_file) as f:
        model_info = pickle.load(f)

    if not os.path.isfile(FLAGS.label_file):
        print('File : ' + FLAGS.label_file + ' does not exist. Aborting!!!')
        print('Aborting!!!')
        return

    # Loading the list of labels
    labels = load_labels(FLAGS.label_file)

    if FLAGS.input_dir:
        inference_from_dir(FLAGS.server, FLAGS.input_dir, FLAGS.model_name, model_info, labels)

    if FLAGS.input_image:
        inference_from_file(FLAGS.server, FLAGS.input_image, FLAGS.model_name, model_info, labels)

    print('Done')


if __name__ == '__main__':
    tf.app.run()
