import numpy as np
#%tensorflow_version 2.x
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
%matplotlib inline

from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator as IDG

from tensorflow.python.keras import backend as K
#K.set_image_dim_ordering('tf')
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


class ResNet:
  @staticmethod
  def id_block(X, size, num_filters):

    shortcut = X

    X = Conv2D(num_filters[0], (1,1), padding='valid',  
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(num_filters[1], (size,size), padding='same',  
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(num_filters[2], (1,1), padding='valid',  
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)

    X = Add()([X, shortcut])
    X = Activation('relu')(X)

    return X

  def conv_block(X, size, num_filters, s):

    shortcut = X

    X = Conv2D(num_filters[0], (1,1), strides=(s,s),  
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(num_filters[1], (size,size), strides=(1,1), padding='same',  
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(num_filters[2], (1,1), strides=(1,1), padding='valid',  
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)

    shortcut = Conv2D(num_filters[2], (1,1), strides=(s,s), padding='valid', 
                      kernel_initializer = glorot_uniform(seed=0))(shortcut)
    shortcut = BatchNormalization(axis=3)(shortcut)

    X = Add()([X, shortcut])
    X = Activation('relu')(X)

    return X


  def build(shape, classes):

    X_input = Input(shape)

    X = ZeroPadding2D((3,3))(X_input)

    X = Conv2D(shape[0], (7,7), strides=(2,2),
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3), strides=(2,2))(X)

    filters = [64,64,256]
    stride = 1

    X = ResNet.conv_block(X, 3, filters, stride)
    for i in range(2):
      X = ResNet.id_block(X, 3, filters)

    filters = [128,128,512]
    stride = 2

    X = ResNet.conv_block(X, 3, filters, stride)
    for i in range(3):
      X = ResNet.id_block(X, 3, filters)

    filters = [256,256,1024]

    X = ResNet.conv_block(X, 3, filters, stride)
    for i in range(5):
      X = ResNet.id_block(X, 3, filters)
      
    filters = [512,512,2048]

    X = ResNet.conv_block(X, 3, filters, stride)
    for i in range(2):
      X = ResNet.id_block(X, 3, filters)

    X = AveragePooling2D((2,2))(X)

    X = Drop

    X = Flatten()(X)
    X = Dense(classes, activation='softmax', 
              kernel_initializer = glorot_uniform(seed=0))(X)
    

    model = Model(input = X_input, outputs=X)

    return model


datagen = IDG(rescale=1. / 255,
    rotation_range=10, 
		fill_mode='nearest',
    vertical_flip= True,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

#'/content/gdrive/My Drive/new/fruits-360/Training'
#'/content/gdrive/My Drive/new/fruits-360/Test'
train_gen = datagen.flow_from_directory('/content/drive/My Drive/freshrotten/dataset/train', 
	target_size=(64, 64), 
	color_mode='rgb',
	batch_size=32,
	class_mode="categorical",
	shuffle=True)

test_gen = datagen.flow_from_directory('/content/drive/My Drive/freshrotten/dataset/test',
    target_size=(64, 64),
		color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
		shuffle=True)

model = ResNet.build((64,64,3),6)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit_generator(train_gen, epochs=30,
    validation_data=test_gen)
