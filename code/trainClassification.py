from trainAbstract import Train
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from modelFactory import ModelFactory
import tensorflow as tf
import pathlib


class TrainClassification(Train):
    def __init__(self, imgPath, name_model="vgg16Depthwise", IMG_SIZE=128, BATCH_SIZE=32, num_classes=3):
        super().__init__(name_model, IMG_SIZE, BATCH_SIZE, num_classes)
        self.imgPath = imgPath
        if name_model == 'googleLeNet' or name_model == 'denseNet':
            self.IMG_SIZE = 224

    def createGenerator(self, imgPath):

        imageGenerator = ImageDataGenerator(brightness_range=[0.5, 1.5],
                                            rotation_range=15,
                                            rescale=1./255,
                                            shear_range=0.2,
                                            vertical_flip=True,
                                            horizontal_flip=True,
                                            fill_mode='nearest'
                                            )
        train_data = imageGenerator.flow_from_directory(directory=imgPath+"/train/", target_size=(
            self.IMG_SIZE, self.IMG_SIZE), color_mode="grayscale", batch_size=self.BATCH_SIZE, class_mode="sparse")
        validation_data = imageGenerator.flow_from_directory(directory=imgPath+"/validation/", target_size=(
            self.IMG_SIZE, self.IMG_SIZE), color_mode="grayscale", batch_size=self.BATCH_SIZE, class_mode="sparse")

        if self.name_model == "googleLeNet":
            def googleLeNetGenerator(generator):
                while True:  # keras requires all generators to be infinite
                    data = next(generator)
                    x = data[0]
                    y = data[1], data[1], data[1]
                    yield x, y

            traindata = googleLeNetGenerator(traindata)
            validationData = googleLeNetGenerator(validationData)

        return train_data, validation_data

    def train(self):

        train_data, validation_data = self.createGenerator(self.imgPath)

        #  BUILD THE MODEL AND TRAIN IT
        modelToUse = ModelFactory.factory(self.name_model)
        model = modelToUse.build(input_shape=(
            self.IMG_SIZE, self.IMG_SIZE, 1), num_classes=self.num_classes)
        print(model.summary())
        print(model.count_params())

        if self.name_model != "googleLeNet":
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                          loss=tf.keras.losses.sparse_categorical_crossentropy,
                          metrics=["accuracy"])
        else:
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                          loss=tf.keras.losses.sparse_categorical_crossentropy,
                          loss_weights={'main': 1.0, 'aux1': 0.3, 'aux2': 0.3},
                          metrics=["accuracy"])

        pathlib.Path('../artifacts/').mkdir(parents=True, exist_ok=True)

        if self.name_model != "googleLeNet":
            save_best_model = tf.keras.callbacks.ModelCheckpoint(
                '../artifacts/best_weights.hdf5', monitor='val_accuracy', verbose=0, save_best_only=True)
        else:
            save_best_model = tf.keras.callbacks.ModelCheckpoint(
                '../artifacts/best_weights.hdf5', monitor='val_main_accuracy', verbose=0, save_best_only=True)

        with tf.device("/gpu:0"):
            history = model.fit(train_data,
                                epochs=300,
                                steps_per_epoch=20,
                                validation_steps=10,
                                validation_data=validation_data,
                                callbacks=[save_best_model])
