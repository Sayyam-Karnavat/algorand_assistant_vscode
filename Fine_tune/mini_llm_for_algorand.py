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