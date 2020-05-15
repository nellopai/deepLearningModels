import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, MaxPool2D, Dropout, Input, GlobalAveragePooling2D, AveragePooling2D, Add, Dense, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


class DenseNet:

    def __init__(self):
        self.dropout_rate = 0.2
        self.nb_filter = 64
        self.growth_rate = 32
        self.compression = 0.5

    def dense_block(self, x, nb_layers):
        concat_feat = x
        for _ in range(nb_layers):
            # 1x1 Convolution (Bottleneck layer)
            inter_channel = self.growth_rate * 4
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(inter_channel, 1, 1, use_bias=False)(x)

            x = Dropout(self.dropout_rate)(x)

            # 3x3 Convolution
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = ZeroPadding2D((1, 1))(x)
            x = Conv2D(self.growth_rate, 3, 1, use_bias=False)(x)

            x = Dropout(self.dropout_rate)(x)
            concat_feat = tf.concat([concat_feat, x], -1)
            self.nb_filter += self.growth_rate
        return concat_feat, self.nb_filter

    def transition_block(self, x):

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(int(self.nb_filter * self.compression), 1, 1,
                   use_bias=False)(x)

        x = Dropout(self.dropout_rate)(x)

        x = AveragePooling2D((2, 2), strides=(2, 2))(x)

        return x

    def build(self, input_shape, num_classes=3):
        denseBlocksNumber = [6, 12, 24, 16]
        activation = 'sigmoid' if num_classes == 1 else 'softmax'
        image = Input(shape=input_shape, name='INPUT_LAYER')
        x = ZeroPadding2D((3, 3))(image)
        x = Conv2D(self.nb_filter, 7, 2, name='conv1', use_bias=False)(image)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = MaxPool2D((3, 3), strides=(2, 2))(x)
        for block_idx in range(len(denseBlocksNumber) - 1):
            x, self.nb_filter = self.dense_block(
                x, denseBlocksNumber[block_idx])
            x = self.transition_block(x)

            self.nb_filter = int(self.nb_filter * self.compression)

        x, self.nb_filter = self.dense_block(x, denseBlocksNumber[-1])

        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        global_average = GlobalAveragePooling2D()(x)
        outputs = Dense(num_classes, activation=activation)(global_average)

        model = Model(inputs=image, outputs=outputs)
        return model
