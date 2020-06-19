from trainFactory import trainFactory
import sys
import predict
import argparse

parser = argparse.ArgumentParser(prog='main',
                                 description='Train a classification model',
                                 conflict_handler='resolve',
                                 prefix_chars='-+')
parser.set_defaults(imgPath=None)
# optional arguments
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-train',
                   dest='train_type', action='store', choices=["classification", "OD"])
group.add_argument('-predict', dest='predict', action='store_true')


known_args = parser.parse_known_args()
if known_args[0].train_type == "classification":
    parser.add_argument('-m', '--modelType', choices=['VGG16GlobalAverage', 'vgg16Depthwise', 'resNet18', 'resNet50', 'googleLeNet', 'denseNet'], default='vgg16Depthwise', type=str, action='store',
                        help="Name of the model to use.")
    parser.add_argument('-imgPath', type=str, required=True, action='store',
                        help="Relative path of the input images.")
elif known_args[0].train_type == "OD":
    parser.add_argument('-m', '--modelType', choices=['RCNN'], default='RCNN', type=str, action='store',
                        help="Name of the model to use.")


args = parser.parse_args()
train_type = args.train_type
predict_flag = args.predict
imgPath = args.imgPath
nameModel = args.modelType

Train = trainFactory.factory(
    train_type, nameModel=nameModel, imgPath=imgPath)
Train.train()

if predict_flag:
    print("TO DO")
