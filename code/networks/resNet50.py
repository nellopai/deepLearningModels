import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, MaxPool2D, Input, GlobalAveragePooling2D, Add, Dense, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


class ResNet50:

    def __init__(self, useZeroPadding):
        self.useZeroPadding = useZeroPadding

    def bottleneck_resNet(self, input_data, filters, conv_size, stride):

        def pad_depth(x, desired_channels):
            new_channels = desired_channels - x.shape.as_list()[-1]
            output = tf.identity(x)
            repetitions = new_channels/x.shape.as_list()[-1]
            for _ in range(int(repetitions)):
                zeroTensors = tf.zeros_like(x, name='pad_depth1')
                output = tf.keras.backend.concatenate([output, zeroTensors])
            return output

        shortcut = input_data
        #
        if (self.useZeroPadding):
            if (shortcut.shape[3] != filters*4):
                shortcut = MaxPool2D(pool_size=(1, 1),
                                     strides=(stride, stride),
                                     padding='same')(shortcut)

                shortcut = tf.keras.layers.Lambda(pad_depth, arguments={
                    'desired_channels': filters*4})(shortcut)
        else:
            shortcut = Conv2D(filters*4, 1, strides=(stride,
                                                     stride), padding='same')(shortcut)

        x = Conv2D(filters, 1, activation=None,
                   padding='same', strides=(stride, stride))(input_data)
        x = BatchNormalization()(x)
        x = Conv2D(filters, conv_size, activation=None, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters*4, 1, activation=None,
                   padding='same')(x)
        x = Add()([shortcut, x])
        x = Activation('relu')(x)
        return x

    def build(self, input_shape, num_classes=3):
        filters = [64, 128, 256, 512]
        activation = 'sigmoid' if num_classes == 1 else 'softmax'
        image = Input(shape=input_shape, name='INPUT_LAYER')

        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),
                       padding="same", activation="relu")(image)

        max_pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(conv1)

        res_block1 = self.bottleneck_resNet(max_pool1, filters[0], 3, 1)
        res_block2 = self.bottleneck_resNet(res_block1, filters[0], 3, 1)
        res_block3 = self.bottleneck_resNet(res_block2, filters[0], 3, 1)

        res_block4 = self.bottleneck_resNet(res_block3, filters[1], 3, 2)
        res_block5 = self.bottleneck_resNet(res_block4, filters[1], 3, 1)
        res_block6 = self.bottleneck_resNet(res_block5, filters[1], 3, 1)
        res_block7 = self.bottleneck_resNet(res_block6, filters[1], 3, 1)

        res_block8 = self.bottleneck_resNet(res_block7, filters[2], 3, 2)
        res_block9 = self.bottleneck_resNet(res_block8, filters[2], 3, 1)
        res_block10 = self.bottleneck_resNet(res_block9, filters[2], 3, 1)
        res_block11 = self.bottleneck_resNet(res_block10, filters[2], 3, 1)
        res_block12 = self.bottleneck_resNet(res_block11, filters[2], 3, 1)
        res_block13 = self.bottleneck_resNet(res_block12, filters[2], 3, 1)

        res_block14 = self.bottleneck_resNet(res_block13, filters[3], 3, 2)
        res_block15 = self.bottleneck_resNet(res_block14, filters[3], 3, 1)
        res_block16 = self.bottleneck_resNet(res_block15, filters[3], 3, 1)

        global_average = GlobalAveragePooling2D()(res_block16)
        outputs = Dense(num_classes, activation=activation)(global_average)

        model = Model(inputs=image, outputs=outputs)
        return model
