# IndoorLocalizationSceneClassifier

## Overview

This project is a part of fully functioning vision based assistive indoor localization system for visually impaired people.
The convolutional neural network was developed to classify high-resolution omnidirectional images to reduce the time complexity of the sample search algorithm. Firstly, the dataset was generated and normalized. Secondly, the CNN’s architecture was developed and global parameters were optimized. Thirdly, the trained model was validated and tested on distinct datasets to ensure generalization
ability correctness. The classification accuracy for top 1 candidate class on the testing dataset is up to 89% on average, for top 2 candidate classes is 100%. All work was done using Tensorflow 1.0 with original Python API. All the insides of the CNN were visualized in the Tensorboard. The results of the research under the lead of Professor Zhigang Zhu will be published in a book “Assistive Computer Vision” by December 2017.

### Architecture

The web application is divided into 6 packages:

1. data - classes and methods for datasets preparation and normalization + convertor of raw image data to .tfrecords format.
2. evaluation - methods for measuring the quality of the model + testing on testing set with confusion matrix output and interactive evaluation.
3. helpers - additional methods.
4. model - classes and methods for building a CNN model.
5. training - methods for training process.
6. visualization - methods for visualizing training and testing processes (activations, kernels etc).

### Motivation

An indoor localization system is of significant importance to the visually impaired in their daily lives if it can help them localize themselves and further navigate unfamiliar indoor environments. There are 285 million visually impaired people in the world according to the World Health Organization, among whom 39 million are blind. Compared to sighted people, it is much harder for visually impaired people to navigate indoor environments. Nowadays, too many buildings are also unfortunately mainly designed and built for sighted people; therefore, navigational tasks and functions that sighted people take for granted could be huge problems to visually impaired people.

## Tech stack

* Python 3.5
* Tensorflow 1.0

### Tools

Pycharm 2016

## Issues

* ...

## Future work

The current trained convolutional neural network is not optimal and it will take many hours to find the optimal architecture and global parameters if we only use CPUs. To speed up things we will train our model on GPU cluster in some cloud like AWS, Azure or Google Cloud Platform.

Second, data collection and manual labeling is still tedious and time consuming, so we will explore efficient strategy to automate this process.

Another important thing is that we work with video stream, which means that captured frames have one crucial fact one more dimension time. This means that each image depends on previous images and influences on the following ones. Thus, for the time series data it is common to use Recurrent Neural Networks, and we will try to combine RNN with our existing approach. Using temporal information also applies to the following steps.
