from keras.layers import (
    Activation, Add, AveragePooling2D, BatchNormalization,
    Concatenate, concatenate, Conv2D, Dense, Dropout, Flatten,
    GlobalAveragePooling2D, Input, MaxPooling2D, ZeroPadding2D
)
from keras import Model
from keras import optimizers
import keras
import numpy as np
import tensorflow as tf
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

data = unpickle('./cifar-10/data_batch_1')
meta = unpickle('./cifar-10/batches.meta')

x = np.array(data[b'data'])
y = np.array(data[b'labels'])
input_shape = (-1, 32, 32, 3)
x = x.reshape(input_shape)

input_shape = (-1, 224, 224, 3)


from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

resnet = ResNet50(weights='imagenet')
resnet.summary()

resnet_json = resnet.to_json()

with open("model.json", "w") as f:
    f.write(resnet_json)

import json

resnet_dict = json.loads(resnet_json)


def conv1_layer():
    def conv1(x_in):
        x = input = Input(shape=x_in.shape[1:])
        x = ZeroPadding2D(padding=(3, 3), name='conv1_pad1')(x)
        x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                   padding='valid', activation=None, name='conv1_conv')(x)
        x = BatchNormalization(name='conv1_bn')(x)
        x = Activation('relu', name='conv1_bn_relu')(x)
        x = ZeroPadding2D(padding=(1, 1), name='conv1_pad2')(x)

        return Model(input, x, name='conv1')(x_in)
    return conv1


def convn_layers(filters1, filters2, filters3, iter, name):
    def conv_block(x_in):

        x = input = Input(x_in.shape[1:])
        shortcut = x

        for i in range(iter):

            if i == 0:
                x = Conv2D(filters=filters1, kernel_size=(1, 1), strides=(2, 2),
                           padding='valid', activation=None, name=name+f'_1_1x1/1_{i+1}')(x)
            else:
                x = Conv2D(filters=filters1, kernel_size=(1, 1), strides=(1, 1),
                           padding='valid', activation=None, name=name+f'_1_1x1/1_{i+1}')(x)
            x = BatchNormalization(name=name+f'_bn1_{i+1}')(x)
            x = Activation('relu', name=name+f'_relu1_{i+1}')(x)

            x = Conv2D(filters=filters2, kernel_size=(3, 3), strides=(1, 1),
                       padding='same', activation=None, name=name+f'_2_3x3/1_{i+1}')(x)
            x = BatchNormalization(name=name+f'_bn2_{i+1}')(x)
            x = Activation('relu', name=name+f'_relu2_{i+1}')(x)

            x = Conv2D(filters=filters3, kernel_size=(1, 1), strides=(1, 1),
                       padding='valid', activation=None, name=name+f'_3_1x1/1_{i+1}')(x)
            x = BatchNormalization(name=name+f'_bn3_{i+1}')(x)

            if i == 0:
                shortcut = Conv2D(filters=filters3, kernel_size=(1, 1), strides=(2, 2),
                                  padding='valid', activation=None, name=name+f'_shortcut_conv_{i+1}')(shortcut)
            else:
                shortcut = Conv2D(filters=filters3, kernel_size=(1, 1), strides=(1, 1),
                                  padding='valid', activation=None, name=name + f'_shortcut_conv_{i + 1}')(shortcut)
            shortcut = BatchNormalization(name=name+f'_shortcut_conv_bn_{i+1}')(shortcut)
            x = Add(name=name+f'_skip_connection_{i+1}')([x, shortcut])
            x = Activation('relu', name=name+f'_skip_relu_{i+1}')(x)

            shortcut = x

        return Model(input, x, name=name)(x_in)

    return conv_block


def resnet50(input_shape, num_classes):

    input = Input(shape=input_shape[1:], dtype='float32', name='input')

    conv1 = conv1_layer()(input)

    conv2_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='conv2_pool')(conv1)

    conv2 = convn_layers(filters1=64, filters2=64, filters3=256, iter=3, name='conv2')(conv2_pool)

    conv3 = convn_layers(filters1=128, filters2=128, filters3=512, iter=4, name='conv3')(conv2)

    conv4 = convn_layers(filters1=256, filters2=256, filters3=1024, iter=6, name='conv4')(conv3)

    conv5 = convn_layers(filters1=512, filters2=512, filters3=2048, iter=3, name='conv5')(conv4)

    gap = GlobalAveragePooling2D()(conv5)

    output = Dense(num_classes, activation='softmax', name='fc_softmax')(gap)

    return Model(input, output, name='resnet50')


input_shape = (-1, 224, 224, 3)

model = resnet50(input_shape=input_shape, num_classes=1000)

model.summary()