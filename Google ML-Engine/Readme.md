# Google ML-Engine
This directory consists of some examples on how to train a CNN model using Google ML-Engine, on how to create the required JSON requests to obtain the prediction of any image, and how to export your model for Android/iOS or for a custom built tensorflow server.

## Description
* ***request/*** :
  * `request_cloud.py` : Python script which can be used to make a prediction from the trained ***Inception-v3*** model deployed on a server using the Google ML-Engine Prediction API.
  * `CNN_client.py` : Python script which can be used to make a prediction from the trained ***Inception-v3*** model deployed on a custom built server using the *tensorflow_serving* library.
* ***trainer/*** :
  * This directory consists of all the codes  required for training the ***Inception-v3*** model on the Google ML-Engine.
* ***utilities/*** :
  * `create_csv.sh` : Bash script which helps in the creation of a csv file required for preprocessing the images before training the model with Google ML-Engine.
  * `freeze_graph.py` : Python script which can be used to export your model in a single file which is suitable for importing it in Android/iOS.
  * `images_to_json.py` : Python script which converts the input images into a JSON request file, which can further be used for getting prediction from the Google ML-Engine Prediction API.

## Example Usage
###### `create_csv.sh`
Before using this script, change the locations of the training dataset and validation dataset directory according to the locations you are going to use. To know about the structure of these directories, refer to the usage of **Trainer Package**.

##### Trainer Package
To learn about how to train the ***Inception-v3*** model on Google ML-Engine, this [tutorial](https://cloud.google.com/blog/big-data/2016/12/how-to-classify-images-with-tensorflow-using-google-cloud-machine-learning-and-cloud-dataflow) is a very good source for the start.

Note that, you have to download and extract a ***inception<...>.tar.gz*** file from this [link](http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz), store the extracted content to some location in your gcloud storage and change the global variable `DEFAULT_INCEPTION_CHECKPOINT` in the `trainer/model.py` script accordingly.

###### `images_to_json.py`
This script can be used to create a JSON request file which can be sent to the Google ML-Engine Prediction API (see the above mentioned tutorial for how to send the JSON request). An example for creating the JSON file for a image, say *abc.jpg* located at */home/xyz/images* :
```sh
python images_to_json.py -o request.json /home/xyz/images/abc.jpg
```

###### `request_cloud.py`
For running this script, you need to install a python library as follows :
```sh
sudo pip install google-cloud-dataflow
```
Also, change the following global variables :
* *desired_width* to the input width of the image accepted by the trained model.
* *desired_height* to the input height of the image accepted by the trained model.
* *PROJECT* to the name of your project on Google Cloud Storage.
* *MODEL* to the name of the model used while deploying it (refer to the above tutorial for how to deploy the model on Google Server API)
* *VERSION* to the version of the model specified while deploying it (refer to the above tutorial for how to deploy the model on Google Server API)
* *LABELS* to the labels your trained model classifies the images into.

Now, to run the script with a single image, say *abc.jpg* (located at */home/xyz/images*), use the script as follows :
```sh
python request_cloud.py --resize --input_image /home/xyz/images/abc.jpg
```
To run with multiple images present in a directory, run :
```sh
python request_cloud.py --resize --input_dir /home/xyz/images
```

###### `CNN_client.py`
Before running the script, change the following global variables :
* *desired_width* to the input width of the image accepted by the trained model.
* *desired_height* to the input height of the image accepted by the trained model.
* *LABELS* to the labels your trained model classifies the images into.

Now, to run the script with a single image, say *abc.jpg* (located at */home/xyz/images*), use the script as follows :
```sh
python CNN_client.py --resize --server=localhost:9000 --input_image /home/xyz/images/abc.jpg --model_name <name of deployed model>
```

To make a request of all images in a directory (say */home/xyz/images*) to your deployed server :
```sh
python CNN_client.py --resize --server=localhost:9000 --input_dir=/home/xyz/images --model_name <name of your deployed server>
```
To know how to start a tensorflow_server, refer to the section **Extra Notes**.

Note that :
1. This script requires the pre-built python libraries stored in *request/tensorflow_serving* directory.
2. You can also follow the guidelines for building your client script using ***bazel*** mentioned in the **Example Usage** of `mobilenet_client.py` inside the MobileNet directory of the main repository. In that case, these pre-built libraries are not required.

###### `freeze_graph.py`
This python script helps in exporting the created model for Android/iOS platforms. But before running the script, keep the following points in mind :
1. Change/Comment the function *os.chdir()* according to your need.
2. Modify the values of the variables *input_graph_path*, *input_binary* and *checkpoint_path* accordingly
3. Modify the variable *output_node_names* with the names of the output nodes of your model. If there are more than one output node, use *commas* to separate their names.

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
