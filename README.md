# Deep Learning Models Repository
This repository includes deep learning models that I implemented.

The **code/** directory contains:

* VGG16 implementation with Global Average Pooling instead of Dense Layers
* VGG16 with Global Average Pooling and MobileNet like structure
* ResNet 18
* ResNet 50
* GoogleLeNet(Inception v1)

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

## How to train a network
In order to train one of the provided networks please run the following command:

python3 main.py -train -i ../data/ -m resNet50

where: 
* -train tells to the main that the user wants to train a network
* -i expects the path where the data is located (read the following section for further details)
* -m expects the name of one of the supported models
    * VGG16GlobalAverage
    * vgg16Depthwise
    * resNet18
    * resNet50
    * googleLeNet

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


One should notice that ResNet18 has less parameter compared to VGG16 without the MobileNet improvement and it converges much faster than VGG16.

 
For a reference to the original paper for VGG please refer to the following link:
https://arxiv.org/abs/1409.1556

For a reference to the original paper for ResNet please refer to the following link:
https://arxiv.org/pdf/1512.03385.pdf

For a reference to the original paper for GoogleLeNet please refer to the following link:
https://arxiv.org/pdf/1409.4842.pdf