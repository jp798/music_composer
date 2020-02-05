import tensorflow as tf
import keras
from keras import layers
from keras.layers import Activation, Dense, Dropout, Conv1D, Conv2D, Flatten, BatchNormalization, ZeroPadding2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from keras.models import Sequential
from keras import regularizers
from keras.optimizers import SGD
import keras.backend as K
from keras.models import load_model
from keras.callbacks import EarlyStopping

# import matplotlib
# %matplotlib inline

## https://github.com/SaewonY/music-genre-classification/blob/master/music_genre.ipynb

import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display
import numpy as np
from numpy import argmax
import pandas as pd
import random

import warnings
warnings.filterwarnings('ignore')


from tensorflow.python.client import device_lib


# Example of Hip-Hop music
y, sr = librosa.load('music_data/data/0/3.wav', duration=10)
ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
print("shape:",ps.shape)

print("array:",ps)






##  -------------Data augmentation (time stretch, pitch shift)  #####
rate = 0.8 

for number in range(1,41):
	if number < 11 :
		y, sr = librosa.load('music_data/data/0/' + str(number) + '.wav')    
	elif number < 21 : 
		y, sr = librosa.load('music_data/data/1/' + str(number) + '.wav')  
	elif number < 31 : 
		y, sr = librosa.load('music_data/data/2/' + str(number) + '.wav')  
	else : 
		y, sr = librosa.load('music_data/data/3/' + str(number) + '.wav')  


	y_changed = librosa.effects.time_stretch(y, rate=rate)
	librosa.output.write_wav('music_data/data_aug1/'+str(number)+'.wav',y_changed, sr)

	print("{}:{}.wav".format(number,number))
    
#Example of Rock music (time-stretch 0.8)
# y, sr = librosa.load('music_data/data/1/12.wav', duration=10)
# ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
# ps.shape

D1 = [] # Dataset1

for number in range(1,41):
	if number < 11 :
		y, sr = librosa.load('music_data/data/0/' + str(number) + '.wav', duration=10)  
		genre = 0
	elif number < 21 : 
		y, sr = librosa.load('music_data/data/1/' + str(number) + '.wav',duration=10)  
		genre = 1
	elif number < 31 : 
		y, sr = librosa.load('music_data/data/2/' + str(number) + '.wav',duration=10) 
		genre = 2
	else : 
		y, sr = librosa.load('music_data/data/3/' + str(number) + '.wav',duration=10) 
		genre = 3
    
	ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
	if ps.shape != (128, 431): continue
	D1.append((ps, genre))
 

D2 = [] # Dataset1

for number in range(1,41):
    
	y, sr = librosa.load('music_data/data_aug1/' + str(number) + '.wav', duration=10)  

	if number < 11 :
		genre = 0
	elif number < 21 : 
		genre = 1
	elif number < 31 : 
		genre = 2
	else : 
		genre = 3
    
	ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
	if ps.shape != (128, 431): continue
	D2.append((ps, genre))

D = D1 + D2
print("Nu|mber of samples: ", len(D))

################ CNN을 활용한 Modeling
dataset = D
random.shuffle(dataset)

#train dev test split 8:1:1
train = dataset[:60]
dev = dataset[60:70]
test = dataset[60:]

X_train, Y_train = zip(*train)
X_dev, Y_dev = zip(*dev)
X_test, Y_test = zip(*test)

# Reshape for CNN input
X_train = np.array([x.reshape( (128, 431, 1) ) for x in X_train])
X_dev = np.array([x.reshape( (128, 431, 1) ) for x in X_dev])
X_test = np.array([x.reshape( (128, 431, 1) ) for x in X_test])

# One-Hot encoding for classes
Y_train = np.array(keras.utils.to_categorical(Y_train, 8))
Y_dev = np.array(keras.utils.to_categorical(Y_dev, 8))
Y_test = np.array(keras.utils.to_categorical(Y_test, 8))


##########
model = Sequential()
input_shape=(128, 431, 1)

model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
model.add(AveragePooling2D((2, 2), strides=(2,2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="same"))
model.add(AveragePooling2D((2, 2), strides=(2,2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="same"))
model.add(AveragePooling2D((2, 2), strides=(2,2)))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(rate=0.5))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(8))
model.add(Activation('softmax'))


model.summary()

epochs = 200
batch_size = 8
learning_rate = 0.01
decay_rate = learning_rate / epochs
momentum = 0.9
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=['accuracy'])

########

tb_hist = keras.callbacks.TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)
hist = model.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=batch_size, validation_data= (X_dev, Y_dev), callbacks=[early_stopping, tb_hist]) 


score = model.evaluate(x=X_test, y=Y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 모델 저장
model.save('music_genre_classification.h5')