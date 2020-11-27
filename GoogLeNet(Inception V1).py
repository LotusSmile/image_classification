from keras.layers import (
    AveragePooling2D, BatchNormalization, Concatenate, concatenate, Conv2D, Dense, Dropout, Flatten,
    GlobalAveragePooling2D, Input, MaxPooling2D
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

def inception_module(f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pool_proj, name):
    def inception(x_in):
        xx = Input(shape=x_in.shape[1:])

        conv_1x1 = Conv2D(filters=f_1x1, kernel_size=(1, 1), strides=(1, 1),
                          padding='same', activation='relu')(xx)

        conv_3x3_reduce = Conv2D(filters=f_3x3_reduce, kernel_size=(1, 1), strides=(1, 1),
                                 padding='same', activation='relu')(xx)

        conv_5x5_reduce = Conv2D(filters=f_5x5_reduce, kernel_size=(1, 1), strides=(1, 1),
                                 padding='same', activation='relu')(xx)

        max_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                padding='same')(xx)

        conv_3x3 = Conv2D(filters=f_3x3, kernel_size=(3, 3), strides=(1, 1),
                          padding='same', activation='relu')(conv_3x3_reduce)

        conv_5x5 = Conv2D(filters=f_5x5, kernel_size=(5, 5), strides=(1, 1),
                          padding='same', activation='relu')(conv_5x5_reduce)

        pool_proj = Conv2D(filters=f_pool_proj, kernel_size=(1, 1), strides=(1, 1),
                           padding='same', activation='relu')(max_pool)
        # concat = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=-1, name=name)
        concat = Concatenate(axis=-1, name=name)([conv_1x1, conv_3x3, conv_5x5, pool_proj])
        return Model(xx, concat, name=name)(x_in)
    return inception


def make_google_net(input_shape):

    xx = Input(shape=input_shape[1:])

    # Layer 1
    conv1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                   padding='same', activation='relu', name='conv1_7x7/2')(xx)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                         padding='same', name='pool1_3x3/2')(conv1)
    batch_norm1 = BatchNormalization()(pool1)

    # Layer 2
    conv2 = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1),
                   padding='same', activation='relu', name='conv2_3x3/1')(batch_norm1)

    pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                         padding='same', name='pool2_3x3/2')(conv2)
    batch_norm2 = BatchNormalization()(pool2)

    # Layer 3
    inception3_a = inception_module(f_1x1=64, f_3x3_reduce=96, f_3x3=128,
                                    f_5x5_reduce=16, f_5x5=32, f_pool_proj=32,
                                    name='inception3_a')(batch_norm2)

    inception3_b = inception_module(f_1x1=128, f_3x3_reduce=128, f_3x3=192,
                                    f_5x5_reduce=32, f_5x5=96, f_pool_proj=64,
                                    name='inception3_b')(inception3_a)

    pool3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                         padding='same', name='pool3_3x3/2')(inception3_b)

    # Layer 4
    inception4_a = inception_module(f_1x1=192, f_3x3_reduce=96, f_3x3=208,
                                    f_5x5_reduce=16, f_5x5=48, f_pool_proj=64,
                                    name='inception4_a')(pool3)

    # Layer 4 Auxiliary Learning 1
    inception4_aux1_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3),
                                            padding='same', name='inception4_aux1_pool')(inception4_a)
    conv4_aux1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1),
                   padding='same', activation='relu', name='conv4_1x1/1_aux1')(inception4_aux1_pool)
    flat4_aux1 = Flatten()(conv4_aux1)
    dense4_aux1 = Dense(units=1024, activation='relu', name='dense4_aux1')(flat4_aux1)
    drop4_aux1 = Dropout(rate=0.7)(dense4_aux1)
    output4_aux1 = Dense(units=10, activation='softmax', name='aux_output1')(drop4_aux1)

    # Layer 4
    inception4_b = inception_module(f_1x1=160, f_3x3_reduce=112, f_3x3=224,
                                    f_5x5_reduce=24, f_5x5=64, f_pool_proj=64,
                                    name='inception4_b')(inception4_a)
    inception4_c = inception_module(f_1x1=128, f_3x3_reduce=128, f_3x3=256,
                                    f_5x5_reduce=24, f_5x5=64, f_pool_proj=64,
                                    name='inception4_c')(inception4_b)
    inception4_d = inception_module(f_1x1=112, f_3x3_reduce=144, f_3x3=288,
                                    f_5x5_reduce=32, f_5x5=64, f_pool_proj=64,
                                    name='inception4_d')(inception4_c)

    # Layer 4 Auxiliary Learning 2
    inception4_aux2_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3),
                                            padding='same', name='inception4_aux2_pool')(inception4_d)
    conv4_aux2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1),
                        padding='same', activation='relu', name='conv4_1x1/1_aux2')(inception4_aux2_pool)
    flat4_aux2 = Flatten()(conv4_aux2)
    dense4_aux2 = Dense(units=1024, activation='relu', name='dense4_aux2')(flat4_aux2)
    drop4_aux2 = Dropout(rate=0.7)(dense4_aux2)
    output4_aux2 = Dense(units=10, activation='softmax', name='aux_output2')(drop4_aux2)

    # Layer 4
    inception4_e = inception_module(f_1x1=256, f_3x3_reduce=160, f_3x3=320,
                                    f_5x5_reduce=32, f_5x5=128, f_pool_proj=128,
                                    name='inception4_e')(inception4_d)
    pool4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                         padding='same', name='pool4_3x3/2')(inception4_e)

    # Layer 5
    inception5_a = inception_module(f_1x1=256, f_3x3_reduce=160, f_3x3=320,
                                    f_5x5_reduce=32, f_5x5=128, f_pool_proj=128,
                                    name='inception5_a')(pool4)
    inception5_b = inception_module(f_1x1=384, f_3x3_reduce=192, f_3x3=384,
                                    f_5x5_reduce=48, f_5x5=128, f_pool_proj=128,
                                    name='inception5_b')(inception5_a)
    pool5 = GlobalAveragePooling2D(name='pool5_global_avg')(inception5_b)
    drop5 = Dropout(rate=0.4)(pool5)
    output = Dense(units=10, activation='softmax', name='output')(drop5)

    return Model(xx, [output, output4_aux1, output4_aux2], name='inception_v1')


google_net = make_google_net(input_shape)

google_net.summary()

google_net.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
                  loss_weights=[1, 0.3, 0.3], optimizer=sgd, metrics=['accuracy'])

history = google_net.fit(x, [y, y, y], validation_split=0.1,
                         epochs=epoch, batch_size=32, callbacks=[lr_sc])

