# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:03:37 2023

@author: Hp
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Embedding, GRU
from keras.utils import pad_sequences

df_full = pd.read_csv('data/PHOENIX-2014-T.train.corpus.csv', sep='|')
df = df_full.drop(columns=['name','video','start','end','speaker'])
data_size = df.shape[1]
#Orth je glossovana recenica, translation je originalna engleska

input_texts = df['translation']
target_texts = df['orth']

#Uradi i analizu karaktera, da li ima cudnih karaktera

target_texts = ['<Start> ' + text + ' <End>' for text in target_texts]

input_texts_split = [text.split() for text in input_texts]
target_texts_split = [text.split() for text in target_texts]

input_words = sorted(set([word for text in input_texts_split for word in text]))
target_words = sorted(set([word for text in target_texts_split for word in text]))

num_input_words = len(input_words)
num_target_words = len(target_words)

input_word_index = {word: ind+1 for ind,word in enumerate(input_words)}
input_word_index[''] = 0
target_word_index = {word: ind+1 for ind,word in enumerate(target_words)}
target_word_index[''] = 0
#0 ce predstavljati praznina zbog paddovanja

inverted_input_word_index = {value: key for key,value in input_word_index.items()}
inverted_target_word_index = {value: key for (key,value) in target_word_index.items()}

max_input_seq_len = np.max([len(text) for text in input_texts_split])
max_target_seq_len = np.max([len(text) for text in target_texts_split])

input_pad_len = max_input_seq_len
target_pad_len = max_target_seq_len 
#Treba ova 2 staviti na vecu vrednost da bi model radio sa duzim recenicama
#Ispada 52 i 32, ovo mi je sumnjivo veliko, istrazi

encoder_input_data = []
for text in input_texts_split:
    encoder_input_data.append([input_word_index.get(word, 0) for word in text])
encoder_input_data = pad_sequences(encoder_input_data, input_pad_len, padding = 'post')

decoder_input_data = []
decoder_output_data = []
for text in target_texts_split:
    decoder_input_data.append([target_word_index.get(word, 0) for word in text])
    decoder_output_data.append([target_word_index.get(word,0) for word in text[1:]]) 
decoder_input_data = pad_sequences(decoder_input_data, target_pad_len, padding = 'post')
decoder_output_data = pad_sequences(decoder_output_data, target_pad_len, padding = 'post')
 

#Model
latent_dim = 512
embedding_size = 300

encoder_input_tensor = Input(shape = (input_pad_len, ))
encoder_embedding = Embedding(input_dim = num_input_words + 1, output_dim = embedding_size, mask_zero = True)(encoder_input_tensor)
_, state_h = GRU(units = latent_dim, return_state = True,  unroll = True)(encoder_embedding)

decoder_input_tensor = Input(shape = (None, ))
decoder_embedding = Embedding(input_dim = num_target_words + 1, output_dim = embedding_size, mask_zero = True)(decoder_input_tensor)
decoder_outputs, _, = GRU(units = latent_dim, return_sequences = True, return_state = True, unroll = False)(decoder_embedding, initial_state = state_h)
output = Dense(units = num_target_words + 1, activation = 'softmax')(decoder_outputs)

model_gru = Model(inputs = [encoder_input_tensor, decoder_input_tensor], outputs = output)
#model.summary()
    
model_gru.compile(loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
batch_size = 64
epochs = 20

history = model_gru.fit([encoder_input_data, decoder_input_data], decoder_output_data, epochs = epochs, batch_size = batch_size, validation_split = 0.1)
#0.853 val_loss, dostignut posle 11 epoha
