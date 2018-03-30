from __future__ import print_function
import numpy as np
import os
import scipy.io as sp
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.utils import plot_model
from keras import backend as K
import matplotlib.pyplot as plt
batch_size = 10
num_classes = 1
epochs = 50


def load_data(filename):
    mat = sp.loadmat(filename)
    mat = mat['sig']
    return mat


def neuron_process(ECG_array, num_neur):
    neur_ECG = np.repeat(ECG_array, num_neur, axis=0)
    return neur_ECG


def concatenate_data(array_list):
    concat_array = []
    for arr in array_list:
        concat_array.append(arr)

    concat_nparr = np.array(concat_array)
    return concat_nparr


def get_training_data(directory):
    files = os.listdir(directory)
    files = [directory + "/" + file for file in files]

    dat_list = []
    for f in files:
        if '.mat' in f:
            dat = load_data(f)
            dat_200 = neuron_process(dat, 200)
            dat_list.append(dat)
            

    dat_array = concatenate_data(dat_list)

    dat_array.reshape((np.shape(dat_array)[0], np.shape(dat_array)[1], np.shape(dat_array)[2], 1))

    return dat_array



# This is LSM 
FolderName = 'LSM out xD/NewNew'
Normal= get_training_data(FolderName+'/Normal')
AFib= get_training_data(FolderName+'/AFib')
Normalize=13
avg_pool_size=40

# #This is Raw ECG
# FolderName='Sample Signals'
# Normal= get_training_data(FolderName+'/Normal_AF_250')
# AFib= get_training_data(FolderName+'/AFib_250')
# Normalize=1/250.0
# avg_pool_size=1


TrainProp=0.5
NormieLen=int(TrainProp*Normal.shape[0])
FibbieLen=int(TrainProp*AFib.shape[0])

x_train = np.concatenate((Normal[0:NormieLen], AFib[0:FibbieLen]))
x_test =  np.concatenate((Normal[NormieLen+1:], AFib[FibbieLen+1:]))
y_train = np.concatenate(([[0] for i in range(NormieLen)], [[1] for i in range(FibbieLen)]))
y_test =  np.concatenate(([[0] for i in range(NormieLen+1,Normal.shape[0])], [[1] for i in range(FibbieLen+1,AFib.shape[0])]))

# x_train=np.divide(x_train, np.max(x_train, 0))
# x_test=np.divide(x_test, np.max(x_test,0))


img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
# input image dimensions




# pixels=x_train[1,:]
# pixels=pixels.reshape((img_rows,img_cols))
# plt.imshow(pixels,cmap='gray')
# plt.show()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0],img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0],img_cols, 1)
    input_shape = (img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train *= Normalize
x_test *= Normalize
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(AveragePooling1D(pool_size=(avg_pool_size),input_shape=input_shape))
model.add(Conv1D(32, kernel_size=(3),
                 activation='relu',
                 input_shape=input_shape,
                 padding='causal'))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Dropout(0.25))
model.add(Conv1D(64, (3), activation='relu'))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.adam(),
              metrics=['accuracy'])




History=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          # validation_split=0)
          validation_data=(x_test,y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

eps = np.arange(epochs)+1

# plot_model(model, to_file='model.eps', show_shapes='True')


plt.figure(1)
plt.subplot(2,1,1)
plt.plot(x_train[0])
plt.title('ECG Normal Input')
plt.subplot(2,1,2)
plt.plot(x_train[-1])
plt.title('ECG AFib Input')
plt.figure(2)
plt.subplot(2,2,1)
plt.plot(eps, History.history['acc'])
plt.title('training acc')
plt.subplot(2,2,2)
plt.plot(eps, History.history['loss'])
plt.title('training loss')
plt.subplot(2,2,3)
plt.plot(eps, History.history['val_acc'])
plt.title('validation acc')
plt.subplot(2,2,4)
plt.plot(eps, History.history['val_loss'])
plt.title('validation loss')
plt.show()