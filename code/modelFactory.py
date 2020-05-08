from networks.vgg16Avg import VGG16GlobalAverage
from networks.vgg16Depthwise import VGG16DwGlobalAverage
from networks.resNet18 import ResNet18


class ModelFactory(object):
    # Create based on class name:
    def factory(type):
        # return eval(type + "()")
        if type == "VGG16GlobalAverage":
            return VGG16GlobalAverage()
        if type == "vgg16Depthwise":
            return VGG16DwGlobalAverage()
        if type == "resNet18":
            return ResNet18()
        assert 0, "Bad model type creation: " + type
    factory = staticmethod(factory)
