# CNN_basic
This directory contains some of the example codes which can be used for getting familiar with the implementation of a CNN model in python using `tensorflow`.

## Description
<!-- * `img_classifier.py` : This python script uses a higher level tensorflow library for implementing a very basic CNN model -->
* `build_image_data.py` : The python script which can be used for creating TFRecords of images for training and evaluation.script.
* `input_image.py` : The python script which takes care of all the processes involved from loading the images from the
	created TFRecords till the creation of batches for training and validation of model.
* `cnn_model.py` : The CNN model which we use is described in this script. It includes all the functions such as for the creation of model, calculating the loss as well as the training step.
* `model_train.py` : This is the front-end of our model, which just uses the `input_image.py` and `cnn_model.py` to train the model. It also handles the logging part required.
* `model_eval.py` : This is the front-end of our model, which just uses the `input_image.py` and `cnn_model.py` to evaluate the model. It also handles the logging part required.

## Example Usage
<!-- ###### `img_classifier.py`
[ ] todo -->

###### `model_train.py` and `model_eval.py`
Both of these scripts can be used for the training as well as the evaluation of a CNN model. To run these scripts, the first step is to build your own dataset of images using the `build_image_data.py` script. An example usage for running the script is :
> python build_image_data.py --train_directory=./train --output_directory=./ --validation_directory=./validate --labels_file=mylabels.txt --train_shards=1 --validation_shards=1 --num_threads=1

Here, the *train\_directory* and the *validation\_directory* consists of the image dataset to be used for training and evaluating the CNN model respectively, and _mylabels.txt_ consists of the names of categories to learn for the image classification. The output of the above command goes inside a directory named as *bin\_data* in your current working directory. To know more about the structure of the *train\_directory* and the *validation\_directory*, please read the initial comments in the script. Also, keep in mind that
1.	The names of the folders inside the provided *train\_directory* and *validation\_directory* must completely match with the names provided in the _mylabels.txt_ file.
2.	The labels created in these TFRecords starts with *1* and not from *0*. This is handled in the function _getImage()_ defined inside the `input_image.py` script, by decrementing the value of the label by 1.

To train the model, run :
```sh
python model_train.py --data_dir ./bin_data
```
This will take some time for the training of the model. After completion, run the following command for evaluating your model :
```sh
python model_eval.py --data_dir ./bin_data
```
By default, the `model_train.py` stores all the files related to your CNN model inside */tmp/cnn\_model* directory, and the `model_eval.py` uses this directory to load the model. To change the output directory of your model, use the flag *train\_dir* as follows :
```sh
python model_train.py --data_dir ./bin_data --train_dir <new_location>
```
And for the evaluation step, use the *checkpoint\_dir* :
```sh
python model_eval.py --data_dir ./bin_data --checkpoint_dir <the new_location>
```
Apart from these flags, there are a lot of other flags which can be passed in the command line for running these scripts. To know more about them, please refer to the
*tf.app.flags.DEFINE\_<data type>()* functions inside the `model_train.py`, `model_eval.py` and `cnn_model.py` scripts.

## Extra Notes
* Since the `cnn_model.py` is not called directly, we can still pass on the command line flags defined inside it in the same way as above. As an example, if you want to change the batch size for your image dataset, run :
```sh
python model_train.py --data_dir ./bin_data --batch_size <new value>
```
* A basic tutorial for using the `build_image_data.py` is available [here](https://agray3.github.io/2016/11/29/Demystifying-Data-Input-to-TensorFlow-for-Deep-Learning.html)
