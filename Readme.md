# Image Classification using CNNs

This repository consists of some example codes for the implementation of Convolutional Neural Networks (CNNs) using Tensorflow, an open source software library for machine learning developed by Google.

## Getting Started

These instructions will help in setting up your environment for the project.

### Prerequisites
The prerequisites for running these programs are
```
Python 2.7.X
Tensorflow
Google Cloud SDK (for training your model using Google ML-Engine)
Bazel (for deploying your own server on a linux machine)
Docker (for deploying your own server on a non-linux machine)
```

#### Installing Tensorflow
To install tensorflow, you can follow the steps mentioned in its [official site](https://www.tensorflow.org/install/)

#### Installing Google Cloud SDK
To install Google Cloud SDK on your machine, follow the steps [here](https://cloud.google.com/sdk/docs/quickstarts).
To set up your environment for working with Cloud Machine Learning Engine, follow the steps [here](https://cloud.google.com/ml-engine/docs/quickstarts/command-line).

#### Installing Bazel
Follow the steps in this [link](https://docs.bazel.build/versions/master/install.html) to install Bazel.

#### Installing Docker
To install Docker, you can follow the steps mentioned in its [official site](https://docs.docker.com/engine/installation/).

## Description of Directories
* **CNN_basic** : This directory contains some basic programs for getting familiar with implementing a CNN model in python using `tensorflow`.
* **Google ML-Engine** : This directory contains the programs for retraining a pre-trained CNN model using Google ML-Engine API.
* **MobileNets** : This directory contains the programs required for retraining various MobileNet CNN models with your custom dataset.

## Some Useful Links
* To get familiar with CNN basics, follow this [link](http://cs231n.github.io/convolutional-networks/). You can also use this [presentation](Introduction to  Convolutional Neural Networks - 30 June 2017.pptx) for getting a summary of this article.
* A [stackoverflow solution](https://stackoverflow.com/questions/42801551/how-do-i-change-the-signatures-of-my-savedmodel-without-retraining-the-model) for changing the signature defs of your saved model without retraining the model.
* To learn about the ***Inception*** model architecture, please refer to this [link](https://hacktilldawn.com/2016/09/25/inception-modules-explained-and-implemented/)
* A tutorial to make your model ready for serving through **tensorflow serving** can be found [here](https://medium.com/towards-data-science/how-to-deploy-machine-learning-models-with-tensorflow-part-1-make-your-model-ready-for-serving-776a14ec3198)
