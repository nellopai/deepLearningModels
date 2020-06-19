import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, GlobalAveragePooling2D, Add, Dense, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


class ResNet18:
    def res_net_block(self, input_data, filters, conv_size, stride):

        if stride == 1:
            shortcut = input_data
        else:
            shortcut = Conv2D(filters, 1, strides=(stride, stride),
                              padding='same')(input_data)

        x = Conv2D(filters, conv_size, activation=None,
                   padding='same', strides=(stride, stride))(input_data)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, conv_size, activation=None, padding='same')(x)
        x = BatchNormalization()(x)

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

        res_block1 = self.res_net_block(max_pool1, filters[0], 3, 1)
        res_block2 = self.res_net_block(res_block1, filters[0], 3, 1)

        res_block3 = self.res_net_block(res_block2, filters[1], 3, 2)
        res_block4 = self.res_net_block(res_block3, filters[1], 3, 1)

        res_block5 = self.res_net_block(res_block4, filters[2], 3, 2)
        res_block6 = self.res_net_block(res_block5, filters[2], 3, 1)

        res_block7 = self.res_net_block(res_block6, filters[3], 3, 2)
        res_block8 = self.res_net_block(res_block7, filters[3], 3, 1)

        global_average = GlobalAveragePooling2D()(res_block8)
        outputs = Dense(num_classes, activation=activation)(global_average)

        model = Model(inputs=image, outputs=outputs)
        return model
