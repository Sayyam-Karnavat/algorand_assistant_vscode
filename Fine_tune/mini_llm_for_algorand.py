import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Activation
from tensorflow.keras.optimizers import RMSprop
import warnings
warnings.filterwarnings('ignore')




filepath = tf.keras.utils.get_file('dataset/qa_pairs.json')

''' 
Reading/converting the file in binary mode
then decoding it to utf-8 to make the file little space efficient for unicode characters
and converting it into lower case to increase the accuray
'''
text = open(filepath,'rb').read().decode('utf-8').lower()


# take only the unique characters from text so that high frequency characters are ignored.

unique_characters = sorted(set(text))


char_to_index = dict((i,c) for c,i in enumerate(unique_characters))

print("character to index : ")
print(char_to_index)


# Also making a dictionary of index value to char for future reference

print("Index to character : ")
index_to_char = dict((c,i) for c,i in enumerate(unique_characters))
print(index_to_char)



sentences = []
next_characters = []

seq_length = 40 
step_size = 3



for i in range(0,len(text)-seq_length,step_size):
    sentences.append(text[i : i + seq_length])# This is our feature data
    next_characters.append(text[i+seq_length])#This is our target data (this statement stores next single character)



x = np.zeros((len(sentences),seq_length,len(unique_characters)), dtype = bool ) # features data 
y = np.zeros((len(sentences),len(unique_characters)) , dtype = bool)



# This loop acts as kind of one-hot encoding of unique characters present in the sentence generated

for i, sentence in enumerate(sentences):
    for t,character in enumerate(sentence):
        x[i,t, char_to_index[character]]= 1
    y[i,char_to_index[next_characters[i]]] = 1 



model = Sequential()
model.add(LSTM(128,input_shape =(seq_length,len(unique_characters))))
model.add(Dense(len(unique_characters)))
model.add(Activation('softmax'))


model.compile(loss = 'categorical_crossentropy' ,optimizer = RMSprop(learning_rate = 0.01))

model.fit(x,y,batch_size = 256 ,epochs = 4)



model.save('text_generator.model')

