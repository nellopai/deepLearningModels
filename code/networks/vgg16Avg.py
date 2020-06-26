import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model


class VGG16GlobalAverage:
    @staticmethod
    def build(input_shape, num_classes=3):

        activation = 'sigmoid' if num_classes == 1 else 'softmax'
        image = Input(shape=input_shape, name='INPUT_LAYER')

        conv1 = Conv2D(filters=64, kernel_size=(3, 3),
                       padding="same", activation="relu")(image)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3),
                       padding="same", activation="relu")(conv1)
        max_pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2)

        conv3 = Conv2D(filters=128, kernel_size=(3, 3),
                       padding="same", activation="relu")(max_pool1)
        conv4 = Conv2D(filters=128, kernel_size=(3, 3),
                       padding="same", activation="relu")(conv3)
        max_pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv4)

        conv5 = Conv2D(filters=256, kernel_size=(3, 3),
                       padding="same", activation="relu")(max_pool2)
        conv6 = Conv2D(filters=256, kernel_size=(3, 3),
                       padding="same", activation="relu")(conv5)
        conv7 = Conv2D(filters=256, kernel_size=(3, 3),
                       padding="same", activation="relu")(conv6)
        max_pool3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv7)

        conv8 = Conv2D(filters=512, kernel_size=(3, 3),
                       padding="same", activation="relu")(max_pool3)
        conv9 = Conv2D(filters=512, kernel_size=(3, 3),
                       padding="same", activation="relu")(conv8)
        conv10 = Conv2D(filters=512, kernel_size=(3, 3),
                        padding="same", activation="relu")(conv9)
        max_pool4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv10)

        conv11 = Conv2D(filters=512, kernel_size=(3, 3),
                        padding="same", activation="relu")(max_pool4)
        conv12 = Conv2D(filters=512, kernel_size=(3, 3),
                        padding="same", activation="relu")(conv11)
        conv13 = Conv2D(filters=512, kernel_size=(3, 3),
                        padding="same", activation="relu")(conv12)
        max_pool5 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv13)

        global_average = GlobalAveragePooling2D()(max_pool5)
        outputs = Dense(num_classes, activation=activation)(global_average)

        model = Model(inputs=image, outputs=outputs)
        return model
