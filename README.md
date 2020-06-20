# Deep Learning Models Repository
This repository includes deep learning models that I implemented.

The **code/** directory contains:
* for classification
    * VGG16 implementation with Global Average Pooling instead of Dense Layers
    * VGG16 with Global Average Pooling and MobileNet like structure
    * ResNet 18
    * ResNet 50
    * GoogleLeNet(Inception v1)
    * DenseNet
* for object detection
    * RCNN
    
The implementations are able to manage 2D images (so images in grayscale).
In particular I got interested into the possibility to reduce the amount of
parameters keeping the generalization power of the network.

## Requirement
Before training any network please install the requirements with the command:
pip3 install -r requirements.txt
The packages required are the following:
- pandas
- pillow
- tqdm
- opencv-python-headless
- tensorflow=2.1.0
- tensorflow-datasets (for object detection)

## How to train a classification network
In order to train one of the provided networks please run the following command:

python3 main.py -train classification -imgPath=../data/  -m resNet18

where: 
* -train tells to the main that the user wants to train a network and expects as an argument *classification*
* -imgPath expects the path where the data is located (read the following section for further details)
* -m expects the name of one of the supported models
    * VGG16GlobalAverage
    * vgg16Depthwise (default)
    * resNet18
    * resNet50
    * googleLeNet
    * denseNet

The path to the data points to a directory where only 2 subdirectories are present:
- train
- validation

In each subdirectories the images are divided per class. The complete structure should be like the following:
```
data
+-- train/
|   +-- class1/
|   +-- class2/
|   +-- class3/
|   +-- class4/
+-- validation/
|   +-- class1/
|   +-- class2/
|   +-- class3/
|   +-- class4/
```

## How to train an object detection network
In order to train one of the provided networks please run the following command:

python3 main.py -train OD
where: 
* -train tells to the main that the user wants to train a network and expects as an argument *OD*
* only model currently available is RCNN

The data used for training and validation is the VOC 2007 that is automatically downloaded by the script. 

## Networks details

A summary of number of parameters for different architectures follows:

Model | Number of Parameters
------------ | -------------
VGG16 with Grayscale input | 165,728,963
VGG16 with Grayscale input and Global Average Pooling| 14,715,075
VGG16 with Grayscale input and Global Average Pooling MobileNet like| 1,414,083
ResNet18 | 11,172,099
ResNet50 with shortcut projection | 38,054,147
ResNet50 with zero padding | 20,721,155
GoogleLeNet(Inception v1) | 7,414,969
DenseNet | 3,202,499

One should notice that ResNet18 has less parameter compared to VGG16 without the MobileNet improvement and it converges much faster than VGG16.

The focus for object detection models is not on the complexity and therefore the number of parameters
is not reported.
In the following table one can find reference papers for each architecture:

Name | Reference Papers
------------ | -------------
Vgg | https://arxiv.org/pdf/1409.1556.pdf
ResNet | https://arxiv.org/pdf/1512.03385.pdf
GoogleLeNet | https://arxiv.org/pdf/1409.4842.pdf
DenseNet | https://arxiv.org/pdf/1608.06993.pdf
Rcnn | https://arxiv.org/pdf/1311.2524.pdf 