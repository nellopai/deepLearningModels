#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
from modelFactory import ModelFactory

# experimental setting usable for GPUs with limited memory
# tf.config.experimental.set_memory_growth(
#    tf.config.list_physical_devices('GPU')[0], True)


def train(nameModel, imgPath):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))

    if (nameModel == 'googleLeNet') or nameModel == 'denseNet':
        IMG_SIZE = 224
    else:
        IMG_SIZE = 128

    BATCH_SIZE = 32

    # here augmentation could be performed

    imageGenerator = ImageDataGenerator(brightness_range=[0.5, 1.5],
                                        rotation_range=15,
                                        rescale=1./255,
                                        shear_range=0.2,
                                        vertical_flip=True,
                                        horizontal_flip=True,
                                        fill_mode='nearest'
                                        )
    traindata = imageGenerator.flow_from_directory(directory=imgPath+"/train/", target_size=(
        IMG_SIZE, IMG_SIZE), color_mode="grayscale", batch_size=BATCH_SIZE, class_mode="sparse")
    validationData = imageGenerator.flow_from_directory(directory=imgPath+"/validation/", target_size=(
        IMG_SIZE, IMG_SIZE), color_mode="grayscale", batch_size=BATCH_SIZE, class_mode="sparse")

    #  BUILD THE MODEL AND TRAIN IT
    modelToUse = ModelFactory.factory(nameModel)
    model = modelToUse.build(input_shape=(
        IMG_SIZE, IMG_SIZE, 1), num_classes=3)
    print(model.summary())
    print(model.count_params())

    def googleLeNetGenerator(generator):
        while True:  # keras requires all generators to be infinite
            data = next(generator)
            x = data[0]
            y = data[1], data[1], data[1]
            yield x, y

    if nameModel != "googleLeNet":
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                      loss=tf.keras.losses.sparse_categorical_crossentropy,
                      metrics=["accuracy"])
    else:
        traindata = googleLeNetGenerator(traindata)
        validationData = googleLeNetGenerator(validationData)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                      loss=tf.keras.losses.sparse_categorical_crossentropy,
                      loss_weights={'main': 1.0, 'aux1': 0.3, 'aux2': 0.3},
                      metrics=["accuracy"])

    pathlib.Path('../artifacts/').mkdir(parents=True, exist_ok=True)

    if nameModel != "googleLeNet":
        save_best_model = tf.keras.callbacks.ModelCheckpoint(
            '../artifacts/best_weights.hdf5', monitor='val_accuracy', verbose=0, save_best_only=True)
    else:
        save_best_model = tf.keras.callbacks.ModelCheckpoint(
            '../artifacts/best_weights.hdf5', monitor='val_main_accuracy', verbose=0, save_best_only=True)

    with tf.device("/gpu:0"):
        history = model.fit(traindata,
                            epochs=300,
                            steps_per_epoch=20,
                            validation_steps=10,
                            validation_data=validationData,
                            callbacks=[save_best_model])

    validation_steps = 20

    loss0, accuracy0 = model.evaluate(validationData, steps=validation_steps)

    print("loss: {:.2f}".format(loss0))
    print("accuracy: {:.2f}".format(accuracy0))
