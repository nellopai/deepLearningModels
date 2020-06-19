from abc import ABC, abstractmethod


class Train(ABC):

    @abstractmethod
    def __init__(self, name_model, IMG_SIZE, BATCH_SIZE, num_classes):
        self.name_model = name_model
        self.IMG_SIZE = IMG_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.num_classes = num_classes

    @abstractmethod
    def createGenerator():
        pass

    @abstractmethod
    def train():
        pass
