from __future__ import print_function

from cStringIO import StringIO
import os
import timeit

from PIL import Image

# This is a placeholder for a Google-internal import.
from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

desired_width = 299
desired_height = 299

LABELS = ['memes', 'quotes', 'others', 'unknown']

tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
tf.app.flags.DEFINE_string('input_dir', '',
                           'A directory containing .jpg or .jpeg files to serialize into a '
                           'request json')
tf.app.flags.DEFINE_string('input_image', '', 'A .jpg image file to be predicted')
tf.app.flags.DEFINE_boolean('resize', True, 'Will resize images locally first.  Not needed, but'
                                            ' will reduce network traffic.')
tf.app.flags.DEFINE_string('model_name', '', 'The name of the model deployed through tensorflow serving')
FLAGS = tf.app.flags.FLAGS


def do_inference(hostport, request_dict, model_name):
    """Tests PredictionService with concurrent requests.

    Args:
        hostport: Host:port address of the PredictionService.
        request_dict: Dictionary for the values to be sent as input to the server.
        model_name: Name of the deployed tensorflow server.
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
    request.inputs['image_bytes'].CopyFrom(
        tf.contrib.util.make_tensor_proto(request_dict['image_bytes']['b64'], shape=[1]))

    start_time = timeit.default_timer()
    res = stub.Predict(request, 27.0)  # 25 seconds
    elapsed = timeit.default_timer() - start_time
    print('Time : ', elapsed)

    key = res.outputs['key'].string_val
    scores = res.outputs['scores'].float_val
    prediction = res.outputs['prediction'].int64_val
    print('Key : ', key)
    print('Prediction Value : ', prediction)
    for i in prediction:
        print('Prediction Label : ', LABELS[i])
    print('Scores : ', scores)


def inference_from_dir(server, input_dir, do_resize, model_name):
    """Makes the request dictionary from all the images in the input_dir

    Args:
        server: The PredictionService host:port
        input_dir: The directory which contains the images.
        do_resize: Boolean specifying if script should resize images.
        model_name: The name of the model deployed on the server.
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
                # Uses argparse to check permissions, but ignore pre-opened file handle.
                image = Image.open(image_handle)
                resized_handle = StringIO()
                is_too_big = ((image.size[0] * image.size[1]) >
                              (desired_width * desired_height))
                if do_resize or is_too_big:
                    image = image.resize((desired_width, desired_height), Image.BILINEAR)

                image.save(resized_handle, format='JPEG')

                # key can be any UTF-8 string, since it goes in a HTTP request.
                request_dict = {'key': image_handle,
                                'image_bytes': {'b64': resized_handle.getvalue()}}

                do_inference(server, request_dict, model_name)

            except IOError as e:
                print('Could not read:', image_handle, ':', e, '- it\'s ok, skipping.')


def inference_from_file(server, image_handle, do_resize, model_name):
    """Makes the request dictionary from the input_image

    Args:
        server: The PredictionService host:port
        image_handle: The name of the image.
        do_resize: Boolean specifying if script should resize images.
        model_name: The name of the model deployed on the server.
    """
    if not os.path.isfile(image_handle):
        print('The input_image ' + image_handle + ' does not exist.')
        print('Aborting!!!')
        exit(1)

    try:
        # Uses argparse to check permissions, but ignore pre-opened file handle.
        image = Image.open(image_handle)
        resized_handle = StringIO()
        is_too_big = ((image.size[0] * image.size[1]) >
                      (desired_width * desired_height))
        if do_resize or is_too_big:
            image = image.resize((desired_width, desired_height), Image.BILINEAR)

        image.save(resized_handle, format='JPEG')

        # key can be any UTF-8 string, since it goes in a HTTP request.
        request_dict = {'key': image_handle,
                        'image_bytes': {'b64': resized_handle.getvalue()}}

        do_inference(server, request_dict, model_name)

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

    if FLAGS.input_dir:
        inference_from_dir(FLAGS.server, FLAGS.input_dir, FLAGS.resize, FLAGS.model_name)

    if FLAGS.input_image:
        inference_from_file(FLAGS.server, FLAGS.input_image, FLAGS.resize, FLAGS.model_name)

    print('Done')


if __name__ == '__main__':
    tf.app.run()
