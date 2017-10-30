# import base64
# import sys
# import json
#
# img = base64.b64encode(open(sys.argv[1], "rb").read())
# print(json.dumps({"key":"1", "image_bytes": {"b64": img}}))

# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Make a request JSON from local images to send to CloudML serving API.
"""

import argparse
import base64
from cStringIO import StringIO
# import json
# import sys
import os

from PIL import Image

import googleapiclient.discovery
from oauth2client.client import GoogleCredentials
# from googleapiclient import errors

desired_width = 299
desired_height = 299

PROJECT = 'hike-analytics-test'
MODEL = 'hike_image_classifier'
VERSION = 'v5'

LABELS = ['memes', 'quotes', 'others', 'unknown']

def parse_args():
    """Handle the command line arguments.

    Returns:
        Output of argparse.ArgumentParser.parse_args.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resize', dest='resize', action='store_true',
                        help='Will resize images locally first.  Not needed, but'
                             ' will reduce network traffic.')
    parser.add_argument('--input_dir', default='',
                        help='A directory containing .jpg or .jpeg files to serialize into a '
                             'request json')
    parser.add_argument('--input_image', default='',
                        help='A directory containing .jpg or .jpeg files to serialize into a '
                             'request json')

    args = parser.parse_args()

    return args


def make_request(request_dict):
    # Getting the default credentials
    credentials = GoogleCredentials.get_application_default()

    # Creating a 'service' object for handling the prediction service
    service = googleapiclient.discovery.build('ml', 'v1', credentials=credentials)
    project_id = 'projects/{}'.format(PROJECT, credentials=credentials)
    # print(project_id)

    name = 'projects/{}/models/{}'.format(PROJECT, MODEL)

    if VERSION is not None:
        name += '/versions/{}'.format(VERSION)

    # print(name)

    # Making the request
    response = service.projects().predict(
        name=name,
        body={'instances': [request_dict]}
    ).execute()
    if 'error' in response:
        raise RuntimeError(response['error'])

    return response


def create_request_json(image_handle, do_resize):
    """Produces a JSON request suitable to send to CloudML Prediction API.

    Args:
        image_handle: string specifying the relative location of the image.
        do_resize: Boolean specifying if script should resize images.
    """

    try:
        # Uses argparse to check permissions, but ignore pre-opened file handle.
        image = Image.open(image_handle)
        resized_handle = StringIO()
        is_too_big = ((image.size[0] * image.size[1]) >
                      (desired_width * desired_height))
        if do_resize and is_too_big:
            image = image.resize((desired_width, desired_height), Image.BILINEAR)

        image.save(resized_handle, format='JPEG')
        encoded_contents = base64.b64encode(resized_handle.getvalue())

        # key can be any UTF-8 string, since it goes in a HTTP request.
        request_dict = {'key': image_handle,
                          'image_bytes': {'b64': encoded_contents}}

        response = make_request(request_dict)

        pred = response['predictions'][0]['prediction']
        key = response['predictions'][0]['key']
        scores = response['predictions'][0]['scores']

        print(LABELS[pred] + '\t\t' + key + '\t\t' + str(scores))

    except IOError as e:
        print('Could not read:', image_handle, ':', e, '- it\'s ok, skipping.')


def reading_input_dir(input_dir, do_resize):
    """Produces a JSON request suitable to send to CloudML Prediction API.

    Args:
        input_dir: string specifying the directory containing images.
        do_resize: Boolean specifying if script should resize images.
    """
    print('PREDICTIONS\tFILE\t\t\t\tSCORES')
    print('-' * 80)

    input_images = os.listdir(input_dir)

    for image in input_images:
        image_handle = os.path.join(input_dir, image)
        create_request_json(image_handle, do_resize)


def main():
    args = parse_args()
    if not args.input_dir and not args.input_image:
        print('Please Specify either the image directory using --image_dir or the image file using --image_file')
        print('Aborting!!!')
        exit(1)

    if args.input_dir and args.input_image:
        print('Please Specify either the image directory or the image file')
        print('Aborting!!!')
        exit(1)

    if args.input_image:
        if not os.path.isfile(args.input_image):
            print('The file ' + args.input_image + ' does not exist.')
            print('Aborting!!!')
            exit(1)

        print('PREDICTIONS\tFILE\t\t\t\tSCORES')
        print('-' * 80)
        create_request_json(args.input_image, args.resize)

    if args.input_dir:
        if not os.path.exists(args.input_dir):
            print('The directory ' + args.input_dir + ' does not exist.')
            print('Aborting!!!')
            exit(1)

        reading_input_dir(args.input_dir, args.resize)


if __name__ == '__main__':
    main()
