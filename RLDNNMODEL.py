"""CLDNNLike model for RadioML.

# Reference:

- [CONVOLUTIONAL,LONG SHORT-TERM MEMORY, FULLY CONNECTED DEEP NEURAL NETWORKS ]

Adapted from code contributed by Mika.
"""
import os

import keras.backend as K
WEIGHTS_PATH = ('resnet_like_weights_tf_dim_ordering_tf_kernels.h5')

from keras.models import Model
from keras.layers import Input,Dense,Conv1D,MaxPool1D,ReLU,Dropout,Softmax,Conv2D
from keras.layers import LSTM,add
from keras.layers import BatchNormalization

def ConvBNReluUnit(input,kernel_size = 8,index = 0):
    x = Conv1D(filters=64, kernel_size=kernel_size, padding='same', activation='relu',kernel_initializer='glorot_uniform',
               name='conv{}'.format(index + 1))(input)
    # x = BatchNormalization(name='conv{}-bn'.format(index + 1))(x)
    # x = MaxPool1D(pool_size=2, strides=2, name='maxpool{}'.format(index + 1))(x)

    return x

def CLDNNLikeModel(weights=None,
             input_shape=[1024,2],
             classes=11,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5  # dropout rate (%)
    tap = 8
    input = Input(input_shape,name='input')
    x = input

    # #Cnvolutional Block
    # L = 3
    # for i in range(L):

    y = ConvBNReluUnit(x,kernel_size =tap,index=1)  #128 2
    x = ConvBNReluUnit(y, kernel_size=tap, index=2) #128 2
    x = ConvBNReluUnit(x, kernel_size=tap, index=3) #128 2

    x=add([x,y])
    print(x.shape)
    #LSTM Unit
    # batch_size,64,2
    x = LSTM(units=128,return_sequences = True,name='LSTM1')(x)

    x = Conv1D(filters=32, kernel_size=4, padding='same', activation='relu',name='CNN4')(x)
    x = LSTM(units=128,name='LSTM2')(x)
    print(x.shape)
    #DNN
    x = Dense(128,activation='selu',name='fc1')(x)
    x = Dropout(dr)(x)
    # x = Dense(128, activation='selu', name='fc2')(x)
    # x = Dropout(dr)(x)
    x = Dense(classes,activation='softmax',name='softmax')(x)

    model = Model(inputs = input,outputs = x)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

import keras
if __name__ == '__main__':
    model = CLDNNLikeModel(None,input_shape=(1024,2),classes=11)

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    print('models layers:', model.layers)
    print('models config:', model.get_config())
    print('models summary:', model.summary())