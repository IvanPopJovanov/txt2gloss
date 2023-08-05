# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:03:37 2023

@author: Hp
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

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
 

embedding_size = 300

input_word_embeddings = {}
target_word_embeddings = {}
with open('glove-embedding/vectors.txt', 'r', encoding = 'utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype = 'float32')
        input_word_embeddings[word] = coefs
        target_word_embeddings[word.upper()] = coefs

oov_words_input = []
for word in input_words:
    if not word in input_word_embeddings:
        oov_words_input.append(word)
#print(len(oov_words_input))
#for word in oov_words_input:
#    print(word)
#270 reci, nekoliko brojeva, vecinom nemacke slozenice

oov_words_target = []
for word in target_words:
    if not word in target_word_embeddings:
        oov_words_target.append(word)
#print(len(oov_words_target))
#for word in oov_words_target:
#    print(word)
#274, end, start + reci sa umlautom + posebne znakovne slozenice - dve reci koje se jednim znakom predstavljaju + znakovi sa klasifikatorom

input_embedding_matrix = np.zeros((num_input_words + 1, embedding_size))
for i in range(num_input_words):
    input_embedding_matrix[i + 1] = input_word_embeddings.get(inverted_input_word_index[i+1], np.zeros(embedding_size))

target_embedding_matrix = np.zeros((num_target_words + 1, embedding_size))
for i in range(num_target_words):
    target_embedding_matrix[i + 1] = target_word_embeddings.get(inverted_target_word_index[i+1], np.zeros(embedding_size))

#Model
#tf.random.set_seed(1)
latent_dim = 512
embedding_size = 300

encoder_input_tensor = Input(shape = (input_pad_len, ))
encoder_embedding = Embedding(input_dim = num_input_words + 1, output_dim = embedding_size, mask_zero = True, weights = [input_embedding_matrix], trainable = True)(encoder_input_tensor)
_, state_h = GRU(units = latent_dim, return_state = True,  unroll = True)(encoder_embedding)

decoder_input_tensor = Input(shape = (target_pad_len, ))
decoder_embedding = Embedding(input_dim = num_target_words + 1, output_dim = embedding_size, mask_zero = True, weights = [target_embedding_matrix], trainable = True)(decoder_input_tensor)
decoder_outputs, _, = GRU(units = latent_dim, return_sequences = True, return_state = True, unroll = True)(decoder_embedding, initial_state = state_h)
output = Dense(units = num_target_words + 1, activation = 'softmax')(decoder_outputs)

model_gru = Model(inputs = [encoder_input_tensor, decoder_input_tensor], outputs = output)
#model.summary()
    
model_gru.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
batch_size = 64
epochs = 20

history = model_gru.fit([encoder_input_data, decoder_input_data], decoder_output_data, epochs = epochs, batch_size = batch_size, validation_split = 0.1)
#0.853 val_loss, dostignut posle 11 epoha
#glove embedding input: 0.813 val_loss, 8 epoha
#glove embedding input fine-tune: 0.795, 8 epoha
#+ glove embedding target fine-tune: 0.798, 6 epoha

epochs_vals = range(0, epochs)
losses=history.history['loss']
val_losses=history.history['val_loss']
plt.title('Losses')
plt.plot(epochs_vals, losses, label='train loss')
plt.plot(epochs_vals, val_losses, label='validation loss')
plt.legend(loc='best')
plt.show()