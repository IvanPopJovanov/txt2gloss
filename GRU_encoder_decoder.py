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
from keras.layers import Input, Dense, Embedding, GRU, Bidirectional, Concatenate, Dropout
from keras.utils import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from nltk.translate.bleu_score import sentence_bleu



def clean_texts(data_frame):
    input_texts = data_frame['translation']
    target_texts = data_frame['orth']
    
    target_texts = ['<Start> ' + text + ' <End>' for text in target_texts]
    target_texts = [text.replace('-', ' - ') for text in target_texts]
    umlaut_dict = {'AE': 'Ä',
                   'OE': 'Ö',
                   'UE': 'Ü'}
    for key in umlaut_dict.keys():
        target_texts = [text.replace(key, umlaut_dict[key]) for text in target_texts]
        
    return [input_texts, target_texts]

def analyse_texts(input_texts, target_texts):
    input_texts_split = [text.split() for text in input_texts]
    target_texts_split = [text.split() for text in target_texts]
    max_input_seq_len = np.max([len(text) for text in input_texts_split])
    max_target_seq_len = np.max([len(text) for text in target_texts_split])
                
    input_words = sorted(set([word for text in input_texts_split for word in text]))
    target_words = sorted(set([word for text in target_texts_split for word in text]))
    
    input_word_counts = {}
    for s in input_texts_split:
        for w in s:
            if w in input_word_counts.keys():
                input_word_counts[w] += 1
            else:
                input_word_counts[w] = 1
    
    target_word_counts = {}
    for s in target_texts_split:
        for w in s:
            if w in target_word_counts.keys():
                target_word_counts[w] += 1
            else:
                target_word_counts[w] = 1
    
    for word in input_words:
        if input_word_counts[word] == 1:
            input_words.remove(word)
    
    for word in target_words:
        if target_word_counts[word] == 1:
            target_words.remove(word)
    
    input_word_index = {word: ind+2 for ind,word in enumerate(input_words)}
    input_word_index[''] = 0
    input_word_index['<Unknown>'] = 1
    target_word_index = {word: ind+2 for ind,word in enumerate(target_words)}
    target_word_index[''] = 0
    target_word_index['<Unknown>'] = 1
    
    return [input_word_index, target_word_index, max_input_seq_len, max_target_seq_len]

def get_embedding_matrices(inverted_input_word_index, inverted_target_word_index):
    num_input_words = len(inverted_input_word_index) - 1
    num_target_words = len(inverted_target_word_index) - 1
    input_word_embeddings = {}
    target_word_embeddings = {}
    with open('glove-embedding/vectors.txt', 'r', encoding = 'utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype = 'float32')
            input_word_embeddings[word] = coefs
            target_word_embeddings[word.upper()] = coefs
    
    input_embedding_matrix = np.zeros((num_input_words + 1, embedding_size))
    for i in range(num_input_words):
        input_embedding_matrix[i + 1] = input_word_embeddings.get(inverted_input_word_index[i+1], np.zeros(embedding_size))

    target_embedding_matrix = np.zeros((num_target_words + 1, embedding_size))
    for i in range(num_target_words):
        target_embedding_matrix[i + 1] = target_word_embeddings.get(inverted_target_word_index[i+1], np.zeros(embedding_size))
    
    return [input_embedding_matrix, target_embedding_matrix]

def create_model_data(input_texts, target_texts, input_word_index, target_word_index, input_pad_len, target_pad_len):
    input_texts_split = [text.split() for text in input_texts]
    target_texts_split = [text.split() for text in target_texts]
   
    encoder_input_data = []
    for text in input_texts_split:
       encoder_input_data.append([input_word_index.get(word, 1) for word in text])
    encoder_input_data = pad_sequences(encoder_input_data, input_pad_len, padding = 'post')

    decoder_input_data = []
    decoder_output_data = []
    for text in target_texts_split:
        decoder_input_data.append([target_word_index.get(word, 1) for word in text])
        decoder_output_data.append([target_word_index.get(word, 1) for word in text[1:]]) 
    decoder_input_data = pad_sequences(decoder_input_data, target_pad_len, padding = 'post', truncating = 'post')
    decoder_output_data = pad_sequences(decoder_output_data, target_pad_len, padding = 'post', truncating = 'post')
    
    return [encoder_input_data, decoder_input_data, decoder_output_data]

df_train = pd.read_csv('data/PHOENIX-2014-T.train.corpus.csv', sep='|')
df_train = df_train.drop(columns=['name','video','start','end','speaker'])
train_size = df_train.shape[0]
#Orth je glossovana recenica, translation je originalna engleska

df_val = pd.read_csv('data/PHOENIX-2014-T.dev.corpus.csv', sep = '|')
df_val.drop(columns = ['name', 'video', 'start', 'end', 'speaker'], inplace = True)
val_size = df_val.shape[0]

df_test = pd.read_csv('data/PHOENIX-2014-T.test.corpus.csv', sep = '|')
df_test.drop(columns = ['name', 'video', 'start', 'end', 'speaker'], inplace = True)
test_size = df_test.shape[0]


#Uradi i analizu karaktera, da li ima cudnih karaktera

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


#oov_words_input = []
#for word in input_words:
    #if not word in input_word_embeddings:
       # oov_words_input.append(word)
#print(len(oov_words_input))
#for word in oov_words_input:
#    print(word)
#270 reci, nekoliko brojeva, vecinom nemacke slozenice

#oov_words_target = []
#for word in target_words:
    #if not word in target_word_embeddings:
       # oov_words_target.append(word)
#print(len(oov_words_target))
#for word in oov_words_target:
#    print(word)
#274, end, start + reci sa umlautom + posebne znakovne slozenice - dve reci koje se jednim znakom predstavljaju + znakovi sa klasifikatorom


embedding_size = 300
input_texts, target_texts = clean_texts(df_train)
input_word_index, target_word_index, max_input_seq_len, max_target_seq_len = analyse_texts(input_texts, target_texts)
input_pad_len = 80
target_pad_len = 60
num_input_words = len(input_word_index) - 1
num_target_words = len(target_word_index) - 1
inverted_input_word_index = {value: key for key,value in input_word_index.items()}
inverted_target_word_index = {value: key for (key,value) in target_word_index.items()}
input_embedding_matrix, target_embedding_matrix = get_embedding_matrices(inverted_input_word_index, inverted_target_word_index)
encoder_input_data, decoder_input_data, decoder_output_data = create_model_data(input_texts, target_texts, input_word_index, target_word_index, input_pad_len, target_pad_len)

input_texts_val, target_texts_val = clean_texts(df_val)
encoder_input_data_val, decoder_input_data_val, decoder_output_data_val = create_model_data(input_texts_val, target_texts_val, input_word_index, target_word_index, input_pad_len, target_pad_len)


#Model
#tf.random.set_seed(1)
latent_dim = 256

encoder_input_tensor = Input(shape = (input_pad_len, ))
encoder_embedding = Embedding(input_dim = num_input_words + 1, output_dim = embedding_size, mask_zero = True, weights = [input_embedding_matrix], trainable = True)(encoder_input_tensor)
_, forward_state, backward_state = Bidirectional(GRU(units = latent_dim, return_state = True, dropout = 0.5))(encoder_embedding)
state_h = Concatenate(axis=-1)([forward_state, backward_state])
#_, state_h = GRU(units = latent_dim, return_state = True, unroll = False)(encoder_embedding)
#encoder_model = Model(encoder_input_tensor, encoder_output)

decoder_input_tensor = Input(shape = (target_pad_len, ))
decoder_embedding = Embedding(input_dim = num_target_words + 1, output_dim = embedding_size, mask_zero = True, weights = [target_embedding_matrix], trainable = True)(decoder_input_tensor)
decoder_outputs, _, = GRU(units = latent_dim*2, return_sequences = True, return_state = True, dropout = 0.5)(decoder_embedding, initial_state = state_h)
output = Dropout(0.5)(decoder_outputs)
output = Dense(units = num_target_words + 1, activation = 'softmax')(decoder_outputs)

model_gru = Model(inputs = [encoder_input_tensor, decoder_input_tensor], outputs = output)
#model_gru.summary()
    
model_gru.compile(optimizer = Adam(0.0002), loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
batch_size = 128
epochs = 5

checkpoint = ModelCheckpoint('best_model_weights.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min')

history = model_gru.fit([encoder_input_data, decoder_input_data], decoder_output_data, epochs = epochs, batch_size = batch_size, validation_data = ([encoder_input_data_val, decoder_input_data_val], decoder_output_data_val), callbacks = [checkpoint])
#0.853 val_loss, dostignut posle 11 epoha
#glove embedding input: 0.813 val_loss, 8 epoha
#glove embedding input fine-tune: 0.795 val_loss, 8 epoha
#+ glove embedding target fine-tune: 0.798 val_loss, 6 epoha
#Mora dosta predprocesiranja da se ubaci da bi embedding potencijalno lepo radio za target
#cistio umlaute u glossovanim tekstovima i crticu odvojio od reci: 0.709 val_loss, 7 epoha 
#Kasnije dobijao oko 0.75 val_loss sa istim setupom?
#Encoder pretvorio u Bidirectional: 0.744 val_loss
#Sada iste performanse ima i sa 256 latent_dim, kompleksnost modela nadoknadjuje manjak dimenzija
#256 latent_dim, dropout 0.5 u svakom gru i posle decoder_outputa, end token popravljen: 0.712
#512 latend_dim, ostalo isto, 0.72 val_loss
#256, promenjen target_pad_len na fiksnih 60, 0.433 val_loss, 36/60 * 0.712 = 0.427 za referencu
#zaseban validacioni, nepoznate reci na praznine, 0.364, praznine ignorise kad racuna loss
#zaseban validacioni, mapiranje reci koje se jednom pojavljuju na <Unknown>

model_gru.load_weights('best_model_weights.h5')

model_gru.save('model_gru_newest.h5')

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
    coded_words =[input_word_index.get(word, 1) for word in words]
    coded_words = pad_sequences([coded_words], maxlen = input_pad_len, padding = 'post')
    #Initialize decoder input with starting token
    decoder_input = np.reshape(target_word_index['<Start>'], (1,1))
    decoder_input = pad_sequences(decoder_input, target_pad_len, padding = 'post') #Pad decoder_input
    #Pass through the whole model sequentially:
    output_sentence = []
    for i in range(target_pad_len):
        decoder_output = model_gru.predict([coded_words, decoder_input], verbose = 0)
        next_word = np.argmax(decoder_output[0, i])
        if next_word == target_word_index['<End>']:
            break
        output_sentence.append(next_word)
        if i < target_pad_len - 1:
            decoder_input[0, i+1] = next_word
    return ' '.join(inverted_target_word_index[num] for num in output_sentence).replace(' - ', '-')

test_input_sentences = df_test['translation']
test_references = df_test['orth']
umlaut_dict = {'AE': 'Ä',
               'OE': 'Ö',
               'UE': 'Ü'}
for key in umlaut_dict.keys():
    test_references = [text.replace(key, umlaut_dict[key]) for text in test_references]
translations = [translate2(input_sentence) for input_sentence in test_input_sentences] #Mnogo sporo, mora da se napravi efikasna verzija
bleu_scores = [sentence_bleu(reference.split(), translation.split()) for reference, translation in zip(test_references, translations)]
bleu_final = np.mean(bleu_scores) #Ne radi lepo, daje samo nule.
    

    
    



               
