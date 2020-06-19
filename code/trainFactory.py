from trainClassification import TrainClassification
from trainOD import TrainOD


class trainFactory(object):
    # Create based on class name:
    def factory(type, nameModel, imgPath=None, IMG_SIZE=128, BATCH_SIZE=32, num_classes=3):
        # return eval(type + "()")
        if type == "classification":
            return TrainClassification(imgPath, nameModel)
        if type == "OD":
            return TrainOD(nameModel)
        assert 0, "Bad train type creation: " + type
    factory = staticmethod(factory)
