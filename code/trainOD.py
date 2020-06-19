#!/usr/bin/env python
# coding: utf-8
from trainAbstract import Train

import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_datasets as tfds
from modelFactory import ModelFactory
from networks.rcnn import RCNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices('GPU')[0], True)


class TrainOD(Train):

    def __init__(self, name_model, IMG_SIZE=224, BATCH_SIZE=20, num_classes=20):
        super().__init__(name_model, IMG_SIZE, BATCH_SIZE, num_classes)

    def createGenerator(self):
        trainList, testList = self.loadTrainingAndTestSetVOC07()
        # all the OD model object should have the capacity to build generators
        # starting from the input VOC07
        train_gen, val_gen = self.model.build_generator(
            self.IMG_SIZE, self.BATCH_SIZE, trainList, testList)
        # else:
        #     NotImplementedError(
        #         "Object Detection model type is not implemented")
        return train_gen, val_gen

    def loadTrainingAndTestSetVOC07(self):
        print("*****************************")
        print("***LOADING TRAINING SET******")
        print("*****************************")
        voc_train = tfds.load(
            "voc/2007", split=tfds.Split.TRAIN, batch_size=-1)
        voc_train = tfds.as_numpy(voc_train)
        # seperate the x and y
        x_train, y_train = voc_train["image"], voc_train["labels"]
        boxes_train = voc_train["objects"]['bbox']
        del (voc_train)

        print("*****************************")
        print("******LOADING TEST SET*******")
        print("*****************************")
        voc_test = tfds.load(
            "voc/2007", split=tfds.Split.VALIDATION, batch_size=-1)
        voc_test = tfds.as_numpy(voc_test)
        x_test, y_test = voc_test["image"], voc_test["labels"]
        boxes_test = voc_test["objects"]['bbox']
        del (voc_test)

        return (x_train, y_train, boxes_train), (x_test, y_test, boxes_test)

    def train(self):
        #  BUILD THE MODEL AND TRAIN IT
        self.model = ModelFactory.factory(self.name_model)
        train_gen, val_gen = self.createGenerator()
        self.model = self.model.build((
            self.IMG_SIZE, self.IMG_SIZE, 1), self.BATCH_SIZE, self.num_classes)

        print("Num GPUs Available: ", len(
            tf.config.experimental.list_physical_devices('GPU')))

        opt = Adam(lr=0.0001)
        self.model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                           optimizer=opt, metrics=["accuracy"])
        self.model.summary()

        save_best_model = tf.keras.callbacks.ModelCheckpoint(
            '../artifacts/best_weights.hdf5', monitor='val_accuracy', verbose=0, save_best_only=True)

        with tf.device("/gpu:0"):
            self.model.fit(
                train_gen,
                steps_per_epoch=20,
                epochs=10,
                validation_data=val_gen,
                validation_steps=10,
                use_multiprocessing=True,
                callbacks=[save_best_model])

        validation_steps = 10

        loss0, accuracy0 = self.model.evaluate(val_gen, steps=validation_steps)

        print("loss: {:.2f}".format(loss0))
        print("accuracy: {:.2f}".format(accuracy0))
