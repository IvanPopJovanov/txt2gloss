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
from keras.layers import Input, Dense, Embedding, GRU, Bidirectional, Concatenate
from keras.utils import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from nltk.translate.bleu_score import sentence_bleu


df_full = pd.read_csv('data/PHOENIX-2014-T.train.corpus.csv', sep='|')
df = df_full.drop(columns=['name','video','start','end','speaker'])
data_size = df.shape[1]
#Orth je glossovana recenica, translation je originalna engleska

input_texts = df['translation']
target_texts = df['orth']

#Uradi i analizu karaktera, da li ima cudnih karaktera

target_texts = ['<Start> ' + text + '<End>' for text in target_texts]

# =============================================================================
# counter = 0
# for text in target_texts:
#     counter += text.count(' -') + text.count('- ')
# print(counter) #0
# 
# counter = 0
# for text in target_texts:
#     counter += text.count('-')
# print(counter) #1611
# #Crtica se cesto pojavljuje u znakovnom i prenosi bitno znacenje, trebalo bi da se tretira kao token
# =============================================================================


target_texts = [text.replace('-', ' - ') for text in target_texts]

umlaut_dict = {'AE': 'Ä',
               'OE': 'Ö',
               'UE': 'Ü'}
for key in umlaut_dict.keys():
    target_texts = [text.replace(key, umlaut_dict[key]) for text in target_texts]

#exceptions_dict = {'AKTÜL': 'AKTUELL'}

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

#Pregled najduzih recenica inputa i targeta
#print(input_texts[np.argmax([len(text) for text in input_texts_split])])
#print(target_texts[np.argmax([len(text) for text in target_texts_split])]) #Ista recenica

input_pad_len = max_input_seq_len
target_pad_len = max_target_seq_len 
#Treba ova 2 staviti na vecu vrednost da bi model radio sa duzim recenicama

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

encoder_input_tensor = Input(shape = (input_pad_len, ))
encoder_embedding = Embedding(input_dim = num_input_words + 1, output_dim = embedding_size, mask_zero = True, weights = [input_embedding_matrix], trainable = True)(encoder_input_tensor)
_, forward_state, backward_state = Bidirectional(GRU(units = latent_dim, return_state = True))(encoder_embedding)
state_h = Concatenate(axis=-1)([forward_state, backward_state])
#_, state_h = GRU(units = latent_dim, return_state = True, unroll = False)(encoder_embedding)
#encoder_model = Model(encoder_input_tensor, encoder_output)

decoder_input_tensor = Input(shape = (target_pad_len, ))
decoder_embedding = Embedding(input_dim = num_target_words + 1, output_dim = embedding_size, mask_zero = True, weights = [target_embedding_matrix], trainable = True)(decoder_input_tensor)
decoder_outputs, _, = GRU(units = latent_dim*2, return_sequences = True, return_state = True)(decoder_embedding, initial_state = state_h)
output = Dense(units = num_target_words + 1, activation = 'softmax')(decoder_outputs)

model_gru = Model(inputs = [encoder_input_tensor, decoder_input_tensor], outputs = output)
#model.summary()
    
model_gru.compile(optimizer = Adam(0.0005), loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
batch_size = 64
epochs = 10

checkpoint = ModelCheckpoint('best_model_weights.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min')

history = model_gru.fit([encoder_input_data, decoder_input_data], decoder_output_data, epochs = epochs, batch_size = batch_size, validation_split = 0.1, callbacks = [checkpoint])
#0.853 val_loss, dostignut posle 11 epoha
#glove embedding input: 0.813 val_loss, 8 epoha
#glove embedding input fine-tune: 0.795 val_loss, 8 epoha
#+ glove embedding target fine-tune: 0.798 val_loss, 6 epoha
#Mora dosta predprocesiranja da se ubaci da bi embedding potencijalno lepo radio za target
#cistio umlaute u glossovanim tekstovima i crticu odvojio od reci: 0.709 val_loss, 7 epoha 
#Kasnije dobijao oko 0.75 val_loss sa istim setupom?
#Encoder pretvorio u Bidirectional: 0.744 val_loss
model_gru.load_weights('best_model_weights.h5')

#model_gru.save('model_gru_newest.h5')

#model_gru.evaluate([encoder_input_data, decoder_input_data], decoder_output_data)

epochs_vals = range(0, epochs)
losses=history.history['loss']
val_losses=history.history['val_loss']
plt.title('Losses')
plt.plot(epochs_vals, losses, label='train loss')
plt.plot(epochs_vals, val_losses, label='validation loss')
plt.legend(loc='best')
plt.show()

def translate2(input_sentence):
    #Code input sentence
    input_sentence = input_sentence.replace('.', '').replace(',', '').replace('!','').replace('"','').replace('?','').lower()
    words = input_sentence.split(' ')
    coded_words =[input_word_index.get(word, 0) for word in words]
    coded_words = pad_sequences([coded_words], maxlen = input_pad_len, padding = 'post')
    #Initialize decoder input with starting token
    decoder_input = np.reshape(target_word_index['<Start>'], (1,1))
    decoder_input = pad_sequences(decoder_input, target_pad_len, padding = 'post') #Pad decoder_input
    #Pass through the whole model sequentially:
    output_sentence = []
    for i in range(target_pad_len):
        decoder_output = model_gru.predict([coded_words, decoder_input], verbose = 0)
        next_word = np.argmax(decoder_output[0, i])
        if next_word == 0:
            break
        output_sentence.append(next_word)
        if i < target_pad_len - 1:
            decoder_input[0, i+1] = next_word
    return ' '.join(inverted_target_word_index[num] for num in output_sentence).replace(' - ', '-')





               
