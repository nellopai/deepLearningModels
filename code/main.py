import train
import sys
import predict
import argparse

parser = argparse.ArgumentParser(prog='main',
                                 description='Train a classification model',
                                 conflict_handler='resolve',
                                 prefix_chars='-+')
# optional arguments
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-train', dest='train', action='store_true')
group.add_argument('-predict', dest='predict', action='store_true')

parser.add_argument('-i', '--imgPath', type=str, required=True, action='store',
                    help="Relative path of the input images.")

parser.add_argument('-m', '--modelType', choices=['VGG16GlobalAverage', 'vgg16Depthwise', 'resNet18', 'resNet50'], default='vgg16Depthwise', type=str, action='store',
                    help="Name of the model to use.")

args = parser.parse_args()

trainFlag = args.train
predictFlag = args.predict
imgPath = args.imgPath
nameModel = args.modelType

if trainFlag:
    train.train(nameModel, imgPath)

if predictFlag:
    print("TO DO")
