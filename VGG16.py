from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, Softmax
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

def make_vgg(input_shape):

    x = Input(shape=input_shape)

    # 1st
    conv1_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
           activation='relu', padding='same')(x)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
           activation='relu', padding='same')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1_2)

    # 2nd
    conv2_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
           activation='relu', padding='same')(pool1)
    conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu', padding='same')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2_2)

    # 3rd
    conv3_1 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
           activation='relu', padding='same')(pool2)
    conv3_2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu', padding='same')(conv3_1)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3_2)

    # 4th
    conv4_1 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu', padding='same')(pool3)
    conv4_2 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu', padding='same')(conv4_1)
    conv4_3 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu', padding='same')(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4_3)

    # 5th
    conv5_1 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu', padding='same')(pool4)
    conv5_2 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu', padding='same')(conv5_1)
    conv5_3 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu', padding='same')(conv5_2)
    pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv5_3)

    # Flatten
    flatten = Flatten()(pool5)

    # FC
    fc6 = Dense(units=4096, activation='relu')(flatten)
    fc7 = Dense(units=4096, activation='relu')(fc6)
    fc8 = Dense(units=1000, activation='softmax')(fc7)

    model = Model(x, fc8, name="vgg")

    return model

vgg = make_vgg(input_shape[1:])

vgg.summary()
initial_lr = 0.01
sgd = optimizers.SGD(lr=initial_lr, momentum=0.9, nesterov=False)
import math
def decay(initial_lr, epoch):
    drop = 0.9
    epoch_drop = 8
    lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epoch_drop))
    return lr

lr_sc = keras.callbacks.LearningRateScheduler(decay, verbose=1)

vgg.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
vgg.fit(x=x, y=y, epochs=100, validation_split=0.2, batch_size=64,
        workers=-1, use_multiprocessing=True)

pred = vgg.predict()


