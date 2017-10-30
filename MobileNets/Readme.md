# MobileNets
This directory contains the codes which can be used for retraining a pre-trained MobileNet CNN Model (developed by Google) and a client code for making the request to a deployed tensorflow server for classifying an image or images.

## Description
* `retrain.py` : The python script which is used to retrain a particular architecture of MobileNet model with your own dataset of images.
* `MobileNet_training.sh` : The bash script which can be used to retrain all the available architectures of MobileNet model, using the retrain.py script.
* `mobilenet_client.py` : The tensorflow_server client which can be used to classify an image or images using a particular MobileNet model deployed on a tensorflow_server.
* `BUILD` : Build file for building the `mobilenet_client.py` through bazel.

## Example Usage
###### `retrain.py`
To retrain a particular architecture, say MobileNet_0.25_128, with your own dataset of images, run the following command
> python retrain.py --image_dir /tmp/data/ --learning_rate 0.01 --testing_percentage 20 --validation_percentage 20 --train_batch_size 32 --validation_batch_size -1 --flip_left_right True --random_scale 30 --random_brightness 30 --eval_step_interval 100 --how_many_training_steps 1000 --architecture mobilenet_0.25_128 --output_graph output_graph.pb --output_dir /tmp/mobilenet --output_labels labels.txt --summaries_dir retrain_logs

This command trains as well as evaluates the specified MobileNet architecture and prints the accuracy of the model on the command line. It stores all the outputs in the specified ***output_dir*** (here */tmp/mobilenet*). To know about the structure of the provided *--image\_dir* , please refer to the initial comments in the script.

The structure of the created ***output_dir*** is as follows :
* ***android/*** :
  * `output_graph.pb` (the name passed to the flag *--output_graph*) : This is the file which can be imported directly in any of the open source available codes for running a tensorflow model on Android/iOS.
* ***final_checkpoint/*** :
  * It consists of the latest checkpoint files created after completion of the training of the model.
* ***retrain_logs/*** :
  * ***train/*** : This directory can be used as a value of _--logdir_ flag while running the `tensorboard` for visualizing the graph and other plots of different variables which were being modified during the course of training of the model.
  * ***validation/*** : This directory can be used as a value of _--logdir_ flag while running the `tensorboard` for visualizing the graph and other plots of different variables which were being modified during the course of evaluation of the model.
* ***server/*** :
  * ***mobilenet/*** : All the files in this directory are required for deploying the trained model on a tensorflow_server. The server deployed with these files will accept the transformed image data (image is converted either on the client side, or on another server) and return the predicted category for the image in the response.
  * ***conversion/*** : All the files in this directory are required for deploying the **Image Conversion Step** on a tensorflow_server. The server deployed with these files will accept the original image in the request, and will send the data of the converted image in the response.
* `labels.txt` : A file which consists of the labels which will be predicted by the model. This file is required as an input to the `mobilenet_client.py` script to give the name of the predicted class.
* `model_info.pickle` : A pickle dump file which is required as an input to the `mobilenet_client.py` script, in order to be able to convert the input image into required format on the client side.


###### `MobileNet_training.sh`
The name of a mobilenet architecture is defined as ***mobilenet_< version\_name>\_< size>***. The possible values for version_name are 1.0, 0.75, 0.50, and 0.25, and for the size are 224, 192, 160, and 128. This makes a total of 16 architectures available for re-training. In order to train all of these architectures, with the same dataset, just to figure out what suits best for your need is quite a repetitive task. This script helps in training all the available mobilenet architectures in a much more organized way.


###### `mobilenet_client.py`
To run this program, the first thing is to clone the tensorflow\_server git repository.
```sh
git clone https://github.com/tensorflow/serving.git
```
Then, copy the *mobilenet_client.py* to any location inside the *tensorflow\_serving* directory. Assuming you have copied it into a newly created directory *tensorflow\_serving/example/mobilenet/*, copy the *BUILD* file into that directory as well, so the final status of the directory becomes
> tensorflow_serving/example/mobilenet/BUILD
> tensorflow_serving/example/mobilenet/mobilenet_client.py

Now, build the client file using :
```sh
bazel build //tensorflow_serving/example/mobilenet:mobilenet_client
```
After the script is built, run the following command to make a request of all images in a directory (say */home/xyz/images*) to your deployed server :
```sh
bazel-bin/tensorflow_serving/example/mobilenet/mobilenet_client --server=localhost:9000 --input_dir=/home/xyz/images --model_info_file /home/xyz/model_info.pickle --label_file /home/xyz/labels.txt --model_name <name of your deployed server>
```
To make the request for a particular image (say */home/xyz/images/abc.jpg*), run :
```sh
bazel-bin/tensorflow_serving/example/mobilenet/mobilenet_client --server=localhost:9000 --input_image=/home/xyz/images/abc.jpg --model_info_file /home/xyz/model_info.pickle --label_file /home/xyz/labels.txt --model_name <name of your deployed server>
```
To know how to start a tensorflow_server, refer to the section **Extra Notes**.

Note that the *model_info.pickle* and the *labels.txt* files are the same files which were created by the `retrain.py` script.

In case you don't want to create a new directory in the repository, and instead use the already existing directory (say *tensorflow_serving/example*), then copy the `mobilenet_client.py` script there and from the BUILD file in this directory, copy the text
```sh
py_binary(
    name = "mobilenet_client",
    srcs = [
        "mobilenet_client.py",
    ],
    deps = [
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
```
and append it inside the BUILD file already present in the *tensorflow_serving/example* directory. Then, build and run this script using :
```sh
bazel build //tensorflow_serving/example:mobilenet_client
bazel-bin/tensorflow_serving/example/mobilenet_client --server=localhost:9000 --input_dir=/home/xyz/images --model_info_file /home/xyz/model_info.pickle --label_file /home/xyz/labels.txt --model_name <name of your deployed server>
```

## Extra Notes
#### Running your own Tensorflow Server
Whether you have a successful *bazel* installation (in a linux machine) or using a docker image for the tensorflow server (in a non-linux machine), the steps for running and deploying the server remains the same. Assuming there exists a directory structure like below :
```
/home/xyz/models/
                1/
                  saved_model.pb
                  variables/
                    variables.data-00000-of-00001
                    variables.index
                2/
                  saved_model.pb
                  variables/
                    variables.data-00000-of-00001
                    variables.index
```
Build the tensorflow_server as follows :
```sh
bazel build //tensorflow_serving/model_servers:tensorflow_model_server
```
After the build is successful, the server can be started as :
```sh
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=<name_of_the_model> --model_base_path=/home/xyz/mobilenet
```
To know more about tensorflow_server, you can refer to the tutorials [here](https://tensorflow.github.io/serving/)
