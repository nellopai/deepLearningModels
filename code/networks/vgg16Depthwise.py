import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, MaxPool2D, Input, GlobalAveragePooling2D, DepthwiseConv2D, Dense, SeparableConv2D, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


class VGG16DwGlobalAverage:
    @staticmethod
    def build(input_shape, num_classes=3):

        activation = 'sigmoid' if num_classes == 1 else 'softmax'
        image = Input(shape=input_shape, name='INPUT_LAYER')

        conv1 = Conv2D(filters=64, kernel_size=(3, 3),
                       padding="same", activation="relu")(image)
        conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3),
                                padding="same", depthwise_regularizer=l2())(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = ReLU()(conv2)
        max_pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2)

        conv3 = Conv2D(filters=128, kernel_size=(3, 3),
                       padding="same")(max_pool1)
        conv3 = BatchNormalization()(conv3)
        conv3 = ReLU()(conv3)
        conv4 = SeparableConv2D(filters=128, kernel_size=(3, 3),
                                padding="same", depthwise_regularizer=l2())(conv3)
        conv4 = BatchNormalization()(conv4)
        conv4 = ReLU()(conv4)
        max_pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv4)

        conv5 = Conv2D(filters=256, kernel_size=(3, 3),
                       padding="same")(max_pool2)
        conv5 = BatchNormalization()(conv5)
        conv5 = ReLU()(conv5)
        conv6 = SeparableConv2D(filters=256, kernel_size=(3, 3),
                                padding="same", depthwise_regularizer=l2())(conv5)
        conv6 = BatchNormalization()(conv6)
        conv6 = ReLU()(conv6)
        conv7 = SeparableConv2D(filters=256, kernel_size=(3, 3),
                                padding="same", depthwise_regularizer=l2())(conv6)
        conv7 = BatchNormalization()(conv7)
        conv7 = ReLU()(conv7)
        max_pool3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv7)

        conv8 = Conv2D(filters=512, kernel_size=(3, 3),
                       padding="same")(max_pool3)
        conv8 = BatchNormalization()(conv8)
        conv8 = ReLU()(conv8)
        conv9 = SeparableConv2D(filters=512, kernel_size=(3, 3),
                                padding="same", depthwise_regularizer=l2())(conv8)
        conv9 = BatchNormalization()(conv9)
        conv9 = ReLU()(conv9)
        conv10 = SeparableConv2D(filters=512, kernel_size=(3, 3),
                                 padding="same", depthwise_regularizer=l2())(conv9)
        conv10 = BatchNormalization()(conv10)
        conv10 = ReLU()(conv10)
        max_pool4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv10)

        conv11 = Conv2D(filters=512, kernel_size=(3, 3),
                        padding="same")(max_pool4)
        conv11 = BatchNormalization()(conv11)
        conv11 = ReLU()(conv11)
        conv12 = SeparableConv2D(filters=512, kernel_size=(3, 3),
                                 padding="same", depthwise_regularizer=l2())(conv11)
        conv12 = BatchNormalization()(conv12)
        conv12 = ReLU()(conv12)
        conv13 = SeparableConv2D(filters=512, kernel_size=(3, 3),
                                 padding="same", depthwise_regularizer=l2())(conv12)
        conv13 = BatchNormalization()(conv13)
        conv13 = ReLU()(conv13)
        max_pool5 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv13)

        global_average = GlobalAveragePooling2D()(max_pool5)
        outputs = Dense(num_classes, activation=activation)(global_average)

        model = Model(inputs=image, outputs=outputs)
        return model
