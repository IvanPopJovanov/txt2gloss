# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:03:37 2023

@author: Hp
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from matplotlib import pyplot as plt


import time
from tensorflow import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Embedding, GRU, Bidirectional, Concatenate, Dropout, Layer, Add
from keras.utils import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import edit_distance

print('proba')

class CustomDropout(Layer):
    def __init__(self, custom_value, rate, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.custom_value = custom_value
        self.rate = rate

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()
    
        if training:
            def dropped_inputs():
                return tf.where(tf.logical_and(
                    tf.random.uniform(shape=tf.shape(inputs)) < self.rate,
                    inputs != 0),  # Masking zeros
                    self.custom_value,
                    inputs)
            return K.in_train_phase(dropped_inputs, inputs, training=training)
        return inputs

        return tf.keras.backend.in_train_phase(dropped_inputs, inputs, training=training)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "custom_value": self.custom_value,
            "rate": self.rate
        })
        return config



def clean_texts(input_texts, target_texts):
    target_texts = ['<Start> ' + text + ' <End>' for text in target_texts]
    target_texts = [text.replace('-', ' - ') for text in target_texts]
    umlaut_dict = {'AE': 'Ä',
                   'OE': 'Ö',
                   'UE': 'Ü'}
    for key in umlaut_dict.keys():
        target_texts = [text.replace(key, umlaut_dict[key]) for text in target_texts]
        
    return [input_texts, target_texts]

def clean_texts_df(data_frame):
    input_texts = data_frame['translation']
    target_texts = data_frame['orth']
    return clean_texts(input_texts, target_texts)

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

#Potencijalno vrati samo jedan, s obzirom da su prakticno kopije, mozda bude bitno za memoriju
def load_embedding_data():
    input_word_embeddings = {}
    target_word_embeddings = {}
    with open('glove-embedding/vectors.txt', 'r', encoding = 'utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype = 'float32')
            input_word_embeddings[word] = coefs
            target_word_embeddings[word.upper()] = coefs
    return [input_word_embeddings, target_word_embeddings]

embedding_size = 300

def get_embedding_matrices(inverted_input_word_index, inverted_target_word_index, input_word_embeddings, target_word_embeddings):
    num_input_words = len(inverted_input_word_index) - 1
    num_target_words = len(inverted_target_word_index) - 1
    
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

df_train_val = pd.concat([df_train, df_val])
df_full = pd.concat([df_train_val, df_test])


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
folds = 5
df_np = df_train_val.to_numpy()
np.random.shuffle(df_np)
total_size = df_np.shape[0]
fold_size = total_size/folds
df_folds = [df_np[int(i*fold_size):int((i+1)*fold_size),] for i in range(folds)]
input_word_embeddings, target_word_embeddings = load_embedding_data()
scores = []
for i in range(folds):
    train_folds = [fold for j, fold in enumerate(df_folds) if j!=i]
    train_folds_pd = [pd.DataFrame(data = fold) for fold in train_folds]
    train_data = pd.concat(train_folds_pd)
    val_data = pd.DataFrame(df_folds[i])
    
    input_texts, target_texts = clean_texts(train_data.iloc[:,1], train_data.iloc[:,0])
    input_word_index, target_word_index, max_input_seq_len, max_target_seq_len = analyse_texts(input_texts, target_texts)
    input_pad_len = 80
    target_pad_len = 60
    num_input_words = len(input_word_index) - 1
    num_target_words = len(target_word_index) - 1
    inverted_input_word_index = {value: key for key,value in input_word_index.items()}
    inverted_target_word_index = {value: key for (key,value) in target_word_index.items()}
    input_embedding_matrix, target_embedding_matrix = get_embedding_matrices(inverted_input_word_index, inverted_target_word_index, input_word_embeddings, target_word_embeddings)
    encoder_input_data, decoder_input_data, decoder_output_data = create_model_data(input_texts, target_texts, input_word_index, target_word_index, input_pad_len, target_pad_len)
    
    input_texts_val, target_texts_val = clean_texts(val_data.iloc[:,1], val_data.iloc[:,0])
    encoder_input_data_val, decoder_input_data_val, decoder_output_data_val = create_model_data(input_texts_val, target_texts_val, input_word_index, target_word_index, input_pad_len, target_pad_len)
    
    #input_texts, target_texts = clean_texts_df(df_train)
    #input_word_index, target_word_index, max_input_seq_len, max_target_seq_len = analyse_texts(input_texts, target_texts)
    #num_input_words = len(input_word_index) - 1
    #num_target_words = len(target_word_index) - 1
    #inverted_input_word_index = {value: key for key,value in input_word_index.items()}
    #inverted_target_word_index = {value: key for (key,value) in target_word_index.items()}
    #input_word_embeddings, target_word_embeddings = load_embedding_data()
    #input_embedding_matrix, target_embedding_matrix = get_embedding_matrices(inverted_input_word_index, inverted_target_word_index, input_word_embeddings, target_word_embeddings)
    #encoder_input_data, decoder_input_data, decoder_output_data = create_model_data(input_texts, target_texts, input_word_index, target_word_index, input_pad_len, target_pad_len)

    #input_texts_val, target_texts_val = clean_texts_df(df_val)
    #encoder_input_data_val, decoder_input_data_val, decoder_output_data_val = create_model_data(input_texts_val, target_texts_val, input_word_index, target_word_index, input_pad_len, target_pad_len)


    #Model
    #tf.random.set_seed(1)

    #def build_and_train_model(latent_dim = 256, dropout_rate = 0.5, custom_dropout_rate = 0.05, embedding_freeze = 30, epochs = 50):
    latent_dim = 256
    dropout_rate = 0.5
    custom_dropout_rate = 0.05

    encoder_input_tensor = Input(shape = (input_pad_len, ))
    modified_input = CustomDropout(1.0, custom_dropout_rate)(encoder_input_tensor)
    encoder_embedding_layer = Embedding(input_dim = num_input_words + 1, output_dim = embedding_size, mask_zero = True, weights = [input_embedding_matrix], trainable = False)
    encoder_embedding = encoder_embedding_layer(modified_input)
    outputs = GRU(units = latent_dim, return_sequences = True, dropout = dropout_rate)(encoder_embedding)
    outputs = Dense(units = embedding_size, activation = 'relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    main_inputs = Add()([encoder_embedding, outputs])
    _, forward_state, backward_state = Bidirectional(GRU(units = latent_dim, return_state = True, dropout = dropout_rate))(main_inputs)
    state_h = Concatenate(axis=-1)([forward_state, backward_state])

    encoder_model = Model(encoder_input_tensor, state_h)

    # =============================================================================
    # decoder_input_tensor = Input(shape = (target_pad_len, ))
    # decoder_embedding = Embedding(input_dim = num_target_words + 1, output_dim = embedding_size, mask_zero = True, weights = [target_embedding_matrix], trainable = True)(decoder_input_tensor)
    # decoder_outputs = GRU(units = latent_dim*2, return_sequences = True, dropout = 0.5)(decoder_embedding, initial_state = state_h)
    # output = Dropout(0.5)(decoder_outputs)
    # output = Dense(units = num_target_words + 1, activation = 'softmax')(decoder_outputs)
    # 
    # model_gru = Model(inputs = [encoder_input_tensor, decoder_input_tensor], outputs = output)
    # model_gru.summary()
    # =============================================================================

    decoder_input_tensor = Input(shape = (target_pad_len, ))
    decoder_embedding_layer = Embedding(input_dim = num_input_words + 1, output_dim = embedding_size, mask_zero = True, weights = [input_embedding_matrix], trainable = False)
    decoder_embedding = decoder_embedding_layer(decoder_input_tensor)
    decoder_GRU_layer = GRU(units = latent_dim*2, return_sequences = True, return_state = True, dropout = dropout_rate)
    decoder_outputs, _ = decoder_GRU_layer(decoder_embedding, initial_state = state_h)
    output = Dropout(dropout_rate)(decoder_outputs)
    decoder_dense_layer = Dense(units = num_target_words + 1, activation = 'softmax')
    output = decoder_dense_layer(output)

    model_gru = Model(inputs = [encoder_input_tensor, decoder_input_tensor], outputs = output)
    #model_gru.summary()

    decoder_input_tensor = Input(shape = (1,))
    decoder_state = Input(shape = (latent_dim*2,))
    decoder_outputs, decoder_state_new = decoder_GRU_layer(decoder_embedding_layer(decoder_input_tensor), initial_state = decoder_state)
    output = decoder_dense_layer(decoder_outputs)

    decoder_model = Model(inputs = [decoder_input_tensor, decoder_state], outputs = [output, decoder_state_new])

    # =============================================================================
    # embedding_indices = [2,-5]
    # embedding_layers = [model_gru.layers[i] for i in embedding_indices]
    # other_layers = [layer for i, layer in enumerate(model_gru.layers) if i not in embedding_indices]
    # 
    # optimizer1 = Adam(learning_rate = 0.00001)
    # optimizer2 = Adam(learning_rate = 0.0005)
    # optimizers_and_layers = [(optimizer1, embedding_layers),(optimizer2, other_layers)]
    # optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    # =============================================================================

    model_gru.compile(optimizer = Adam(0.001), loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
    batch_size = 128
    epochs = 10
    embedding_freeze = 0

    checkpoint = ModelCheckpoint('best_model_weights.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min')

    history = model_gru.fit([encoder_input_data, decoder_input_data], decoder_output_data, epochs = embedding_freeze, batch_size = batch_size, validation_data = ([encoder_input_data_val, decoder_input_data_val], decoder_output_data_val), callbacks = [checkpoint])

    if embedding_freeze!=0:
        model_gru.load_weights('best_model_weights.h5')
    decoder_embedding_layer.trainable = True
    encoder_embedding_layer.trainable = True
    #model_gru.summary()
    model_gru.compile(optimizer = Adam(0.0005), loss = 'sparse_categorical_crossentropy', metrics = ['acc'])

    history = model_gru.fit([encoder_input_data, decoder_input_data], decoder_output_data, epochs = epochs - embedding_freeze, batch_size = batch_size, validation_data = ([encoder_input_data_val, decoder_input_data_val], decoder_output_data_val), callbacks = [checkpoint])
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
    #zaseban validacioni, mapiranje reci koje se jednom pojavljuju na <Unknown>, 0.364 val_loss
    #nakon 0.05 dropouta na <Unknown> nad inputima, 0.358 val_loss, sporije dosta konvergira, treba oko 80 epoha
    #dodao jos jedan GRU na pocetak encodera, sa rezidualnom konekcijom na sledeci, 0.368 val_loss
    #dodao dropout na prvi GRU, dodao gust sloj i dropout posle prvog GRU-a, 0.353 val_loss (nije iz prve, vec tek sa default learning_rateom)
    #Smanjio learning_rate za embedding slojeve, probao i da ih zaledim neki broj epoha, ali nije unapredilo model
    
    model_gru.load_weights('best_model_weights.h5')
    
    model_gru.save('model_gru_newest.h5')

    epochs_vals = range(0, epochs - embedding_freeze)
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

    #Brzi translate
    def translate3(input_sentences):
        input_sentences = [input_sentence.replace('.', '').replace(',', '').replace('!','').replace('"','').replace('?','').lower() for input_sentence in input_sentences]
        data_length = len(input_sentences)
        target_placeholder_sentences = ['']*data_length
        encoder_input_data, _, _ = create_model_data(input_sentences, target_placeholder_sentences, input_word_index, target_word_index, input_pad_len, target_pad_len)
        decoder_input_data = np.zeros((data_length, target_pad_len))
        decoder_input_data[:, 0] = np.zeros(data_length) + target_word_index['<Start>']
        decoder_output = np.zeros((data_length, input_pad_len))
        for i in range(target_pad_len-1):
            decoder_output = model_gru.predict([encoder_input_data, decoder_input_data], verbose = 0)
            next_words = np.argmax(decoder_output[:, i, :], axis = -1)
            decoder_input_data[:, i+1] = next_words 
        output_sentences = [' '.join(inverted_target_word_index[num] for num in output_sentence) for output_sentence in decoder_input_data[:,1:]]
        output_sentences = [output_sentence.split('<End>',1)[0].replace(' - ','-') for output_sentence in output_sentences]
        return output_sentences

    #Jos brzi
    def translate4(input_sentences):
        input_sentences = [input_sentence.replace('.', '').replace(',', '').replace('!','').replace('"','').replace('?','').lower() for input_sentence in input_sentences]
        data_length = len(input_sentences)
        target_placeholder_sentences = ['']*data_length
        encoder_input_data, _, _ = create_model_data(input_sentences, target_placeholder_sentences, input_word_index, target_word_index, input_pad_len, target_pad_len)
        encoder_final_state = encoder_model.predict(encoder_input_data, verbose = 0)
        decoder_input_data = np.zeros((data_length, 1)) + target_word_index['<Start>']
        decoder_state = encoder_final_state
        decoder_output = np.zeros((data_length, target_pad_len - 1))
        for i in range(target_pad_len - 1):
            decoder_output_temp, decoder_state = decoder_model.predict([decoder_input_data, decoder_state], verbose = 0)
            next_words = np.argmax(decoder_output_temp, axis = -1)
            decoder_input_data = next_words
            decoder_output[:, i] = next_words.reshape((data_length,))
        output_sentences = [' '.join(inverted_target_word_index[num] for num in output_sentence) for output_sentence in decoder_output]
        output_sentences = [output_sentence.split('<End>',1)[0].replace(' - ','-') for output_sentence in output_sentences]
        return output_sentences

    def evaluate(test_input_sentences, test_references):
        umlaut_dict = {'AE': 'Ä',
                        'OE': 'Ö',
                        'UE': 'Ü'}
        for key in umlaut_dict.keys():
            test_references = [text.replace(key, umlaut_dict[key]) for text in test_references]
        translations = translate4(test_input_sentences)
        smooth_bleu4 = np.mean([sentence_bleu([reference.split()], translation.split(), smoothing_function = SmoothingFunction().method7) for reference, translation in zip(test_references, translations)])
        smooth_bleu3 = np.mean([sentence_bleu([reference.split()], translation.split(), smoothing_function = SmoothingFunction().method7, weights = [1/3, 1/3, 1/3]) for reference, translation in zip(test_references, translations)])
        smooth_bleu2 = np.mean([sentence_bleu([reference.split()], translation.split(), smoothing_function = SmoothingFunction().method7, weights = [1/2, 1/2]) for reference, translation in zip(test_references, translations)])
        smooth_bleu1 = np.mean([sentence_bleu([reference.split()], translation.split(), smoothing_function = SmoothingFunction().method7, weights = [1]) for reference, translation in zip(test_references, translations)])
        #Smoothing je preporucen za recenice, posebno kratke, a glossovane recenice prirodno imaju manji broj reci
        wer = np.mean([edit_distance(reference.split(), translation.split())/len(reference.split()) for reference, translation in zip(test_references, translations)])
        return [wer, smooth_bleu4, smooth_bleu3, smooth_bleu2, smooth_bleu1]

    score = evaluate(input_texts_val, target_texts_val)
    print(score)
    scores.append(score)
    
    



   

#Prilagodi evaluate i ostale funkcije da moze da im se prosledi model, da bi ovde mogle da se iskoriste
#def cv_evaluate(df_train_val, folds = 10): 


# =============================================================================
# start_time = time.time()
# translation2 = [translate2(input_sentence) for input_sentence in test_input_sentences]
# end_time = time.time()
# print("Elapsed time for translation 2: ", end_time - start_time) #612s
# 
# start_time = time.time()
# translation3 = translate3(test_input_sentences)
# end_time = time.time()
# print("Elapsed time for translation 3: ", end_time - start_time) #96s
#
# start_time = time.time()
# translation4 = translate4(test_input_sentences)
# end_time = time.time()
# print("Elapsed time for translation 3: ", end_time - start_time) #15s
# =============================================================================


