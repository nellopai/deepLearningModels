import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Concatenate, Conv2D, MaxPool2D, Input, AveragePooling2D, Add, Dense, Flatten, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


class GoogleLeNet:

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def inception(self, x, filters):
        # 1x1
        path1 = Conv2D(filters=filters[0], kernel_size=(
            1, 1), strides=1, padding='same', activation='relu')(x)

        # 1x1->3x3
        path2 = Conv2D(filters=filters[1][0], kernel_size=(
            1, 1), strides=1, padding='same', activation='relu')(x)
        path2 = Conv2D(filters=filters[1][1], kernel_size=(
            3, 3), strides=1, padding='same', activation='relu')(path2)

        # 1x1->5x5
        path3 = Conv2D(filters=filters[2][0], kernel_size=(
            1, 1), strides=1, padding='same', activation='relu')(x)
        path3 = Conv2D(filters=filters[2][1], kernel_size=(
            5, 5), strides=1, padding='same', activation='relu')(path3)

        # 3x3->1x1
        path4 = MaxPool2D(pool_size=(3, 3), strides=1, padding='same')(x)
        path4 = Conv2D(filters=filters[3], kernel_size=(
            1, 1), strides=1, padding='same', activation='relu')(path4)

        return Concatenate(axis=-1)([path1, path2, path3, path4])

    def auxiliaryNetwork(self, x, name=None):
        layer = AveragePooling2D(
            pool_size=(5, 5), strides=3, padding='valid')(x)
        layer = Conv2D(filters=128, kernel_size=(1, 1), strides=1,
                       padding='same', activation='relu')(layer)
        layer = Flatten()(layer)
        layer = Dense(units=256, activation='relu')(layer)
        layer = Dropout(0.4)(layer)
        layer = Dense(units=self.num_classes,
                      activation='softmax', name=name)(layer)
        return layer

    def build(self, input_shape, num_classes=3):
        self.num_classes = num_classes
        activation = 'sigmoid' if num_classes == 1 else 'softmax'
        layer_in = Input(shape=input_shape, name='INPUT_LAYER')

        # stage-1
        layer = Conv2D(filters=64, kernel_size=(7, 7), strides=2,
                       padding='same', activation='relu')(layer_in)
        layer = MaxPool2D(pool_size=(3, 3), strides=2,
                          padding='same')(layer)
        layer = BatchNormalization()(layer)

        # stage-2
        layer = Conv2D(filters=64, kernel_size=(1, 1), strides=1,
                       padding='same', activation='relu')(layer)
        layer = Conv2D(filters=192, kernel_size=(3, 3), strides=1,
                       padding='same', activation='relu')(layer)
        layer = BatchNormalization()(layer)
        layer = MaxPool2D(pool_size=(3, 3), strides=2,
                          padding='same')(layer)

        # stage-3
        layer = self.inception(layer, [64,  (96, 128), (16, 32), 32])  # 3a
        layer = self.inception(layer, [128, (128, 192), (32, 96), 64])  # 3b
        layer = MaxPool2D(pool_size=(3, 3), strides=2,
                          padding='same')(layer)

        # stage-4
        layer = self.inception(layer, [192,  (96, 208),  (16, 48),  64])  # 4a
        aux1 = self.auxiliaryNetwork(layer, name='aux1')
        layer = self.inception(layer, [160, (112, 224),  (24, 64),  64])  # 4b
        layer = self.inception(layer, [128, (128, 256),  (24, 64),  64])  # 4c
        layer = self.inception(layer, [112, (144, 288),  (32, 64),  64])  # 4d
        aux2 = self.auxiliaryNetwork(layer, name='aux2')
        layer = self.inception(layer, [256, (160, 320), (32, 128), 128])  # 4e
        layer = MaxPool2D(pool_size=(3, 3), strides=2,
                          padding='same')(layer)

        # stage-5
        layer = self.inception(layer, [256, (160, 320), (32, 128), 128])  # 5a
        layer = self.inception(layer, [384, (192, 384), (48, 128), 128])  # 5b
        layer = AveragePooling2D(pool_size=(
            7, 7), strides=1, padding='valid')(layer)

        # stage-6
        layer = Flatten()(layer)
        layer = Dropout(0.4)(layer)
        layer = Dense(units=256, activation='linear')(layer)
        main = Dense(units=self.num_classes, activation=activation,
                     name='main')(layer)

        model = Model(inputs=layer_in, outputs=[main, aux1, aux2])

        return model
