# Classification Models Repository
This repository includes classification models that I implemented.

The **code/** directory contains:

* VGG16 implementation with Global Average Pooling instead of Dense Layers
* VGG16 with Global Average Pooling and MobileNet like structure

The implementations are able to manage 2D images (so images in grayscale).
In particular I got interested into the possibility to reduce the amount of
parameters keeping the generalization power of the network.

A summary of number of parameters for different architectures follows:

Model | Number of Parameters
------------ | -------------
VGG16 with Grayscale input | 165,728,963
VGG16 with Grayscale input and Global Average Pooling| 14,715,075
VGG16 with Grayscale input and Global Average Pooling Mobilenet like| 1,414,083

For a reference to the original paper for VGG please refer to the following link:
https://arxiv.org/abs/1409.1556