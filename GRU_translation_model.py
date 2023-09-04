# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 11:45:52 2023

@author: Hp
"""
#Neretko se u toku prevodjenja javlja greska da je nestalo memorije, ovo se u gresci preporucuje:
#import os
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from matplotlib import pyplot as plt

import os
import time
from tensorflow import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Embedding, GRU, Bidirectional, Concatenate, Dropout, Layer, Add, LayerNormalization
from keras.utils import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import edit_distance

#Mozda resi problem sa memorijom GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

checkpoint = ModelCheckpoint('model_weights_{epoch}.h5', save_best_only=False, save_weights_only=True, monitor='val_loss', mode='min')
#Ckeckpoint se vise ne koristi
early_stopping = EarlyStopping(patience = 5, restore_best_weights = True, monitor = 'val_loss', mode = 'min', verbose = 1)


embedding_size = 300

#Dropout koji umesto nule, menja inpute sa zadatom vrednoscu
#Takodje ne reskalira ostale inpute
#Primena je na tekstualnom inputu, da se reci nasumicno zamene sa <Unknown> tokenom, radi boljeg procesiranja nepoznatih reci
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

#Cisti zadate input i target tekstove
#Dodaje <Start> i <End> tokene
#Pravilno upisuje posebna Nemacka slova
#Takodje za Glossove crtu odvaja od reci koje spaja, da bi se tretirala kao poseban token
def clean_texts(input_texts, target_texts):
    target_texts = ['<Start> ' + text + ' <End>' for text in target_texts]
    target_texts = [text.replace('-', ' - ') for text in target_texts]
    umlaut_dict = {'AE': 'Ä',
                   'OE': 'Ö',
                   'UE': 'Ü'}
    for key in umlaut_dict.keys():
        target_texts = [text.replace(key, umlaut_dict[key]) for text in target_texts]
        
    return [input_texts, target_texts]

#Wrapper za clean_texts koji se direktno da dataframe primenjuje
def clean_texts_df(data_frame):
    input_texts = data_frame['translation']
    target_texts = data_frame['orth']
    return clean_texts(input_texts, target_texts)

#Prolazi kroz tekstove i izdvaja iz njih recnik. Reci koje se pojavljuju samo jednom se ignorisu, kasnije se mapiraju na poseban <Unknown> token
#Takodje belezi maksimalne duzine recenica za input i target
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


#Ucitava pretrenirane glove embedding vektore
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



#Mapira reci iz recnika na odgovarajuce pretrenirane glove embeddinge
#Reci koje nemaju odgovarajuci embedding dobijaju nula vektor kao embedding
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

#Ucitava pretrenirane embedding vektora i mapira reci iz recnika na njih
#U sustini wrapper za load_embedding_data i get_embedding_matrices
#Prednost je sto ce ovako odmah nakon zavrsetka funkcije da se oslobodi memorije za sve embeddinge, koji zauzimaju oko 6GB
def load_embedding_data_get_matrices(inverted_input_word_index, inverted_target_word_index):
    input_word_embeddings = {}
    target_word_embeddings = {}
    with open('glove-embedding/vectors.txt', 'r', encoding = 'utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype = 'float32')
            input_word_embeddings[word] = coefs
            target_word_embeddings[word.upper()] = coefs
            
    num_input_words = len(inverted_input_word_index) - 1
    num_target_words = len(inverted_target_word_index) - 1
    
    input_embedding_matrix = np.zeros((num_input_words + 1, embedding_size))
    for i in range(num_input_words):
        input_embedding_matrix[i + 1] = input_word_embeddings.get(inverted_input_word_index[i+1], np.zeros(embedding_size))

    target_embedding_matrix = np.zeros((num_target_words + 1, embedding_size))
    for i in range(num_target_words):
        target_embedding_matrix[i + 1] = target_word_embeddings.get(inverted_target_word_index[i+1], np.zeros(embedding_size))
    
    return [input_embedding_matrix, target_embedding_matrix]

#Iz recenica kreira podatke koji su spremni da se unose u model
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

#Embedding size, input_pad_len i target_pad_len su fiksni
#Dropout se primenjuje na vise mesta, i svuda je isti dropout_rate
#Na samom pocetku modelu se primenjuje custom_dropout, kako bi naucio da radi sa <Unknown> tokenom bolje
#Encoder ima 3 GRU sloja, poslendji je dvosmeran; izmedju svaka 2 postoje rezidualne veze
#Posto je poslednji sloj decodera dvosmeran GRU, latentna dimenzija dekodera je duplo veca
#Deocoder ima 2 GRU sloja, izmedju postoje rezidualne veze
#Metoda translate sekvencijalni prevodi podatke rec po rec, zbog toga poziva dekoder onoliko puta, koliko je maksimalna duzina target recenice
class GRU_Translation_Model(Model):
    def __init__(self, num_input_words, num_target_words, input_embedding_matrix, target_embedding_matrix, latent_dim = 256, dropout_rate = 0.5, custom_dropout_rate = 0.05):
        super(GRU_Translation_Model, self).__init__()
        
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.custom_dropout_rate = custom_dropout_rate
        self.num_input_words = num_input_words
        self.num_target_words = num_target_words
        self.input_embedding_matrix = input_embedding_matrix
        self.target_embedding_matrix = target_embedding_matrix
        self.embedding_size = 300
        self.input_pad_len = 80
        self.target_pad_len = 60
        
        encoder_input_tensor = Input(shape = (self.input_pad_len, ))
        modified_input = CustomDropout(1.0, custom_dropout_rate)(encoder_input_tensor)

        encoder_embedding_layer = Embedding(input_dim = num_input_words + 1, output_dim = self.embedding_size, mask_zero = True, weights = [input_embedding_matrix], trainable = True)
        encoder_embedding = encoder_embedding_layer(modified_input)
        outputs = GRU(units = latent_dim, return_sequences = True, dropout = dropout_rate)(encoder_embedding)
        outputs = Dense(units = self.embedding_size, activation = 'relu')(outputs)
        outputs = LayerNormalization()(outputs)
        #outputs = Dropout(dropout_rate)(outputs)
        next_inputs = Add()([encoder_embedding, outputs])
        outputs = GRU(units = latent_dim, return_sequences = True, dropout = dropout_rate)(next_inputs)
        outputs = Dense(units = self.embedding_size, activation = 'relu')(outputs)
        outputs = LayerNormalization()(outputs)
        main_inputs = Add()([next_inputs, outputs])
        main_inputs = LayerNormalization()(main_inputs)
        _, forward_state, backward_state = Bidirectional(GRU(units = latent_dim, return_state = True, dropout = dropout_rate))(main_inputs)
        state_h = Concatenate(axis=-1)([forward_state, backward_state])

        self.encoder = Model(encoder_input_tensor, state_h)
        
        decoder_input_tensor = Input(shape = (None, ))
        decoder_starting_state = Input(shape = (latent_dim*2,))
        decoder_embedding_layer = Embedding(input_dim = num_target_words + 1, output_dim = self.embedding_size, mask_zero = True, weights = [target_embedding_matrix], trainable = True)
        decoder_embedding = decoder_embedding_layer(decoder_input_tensor)
        decoder_outputs = GRU(units = latent_dim*2, return_sequences = True, return_state = False, dropout = dropout_rate)(decoder_embedding, initial_state = decoder_starting_state)
        decoder_outputs = Dense(units = self.embedding_size, activation = 'relu')(decoder_outputs)
        main_inputs = Add()([decoder_embedding, decoder_outputs])
        main_inputs = LayerNormalization()(main_inputs)
        decoder_outputs, decoder_state = GRU(units = latent_dim*2, return_sequences = True, return_state = True, dropout = dropout_rate)(main_inputs, initial_state = decoder_starting_state) 
        output = Dropout(dropout_rate)(decoder_outputs)
        output = Dense(units = num_target_words + 1, activation = 'softmax')(output)
        
        self.decoder = Model([decoder_input_tensor, decoder_starting_state], [output, decoder_state])
        
    def call(self, x):
        encoder_output = self.encoder(x[0])
        decoder_output,_ = self.decoder([x[1],encoder_output])
        return decoder_output
    
    #Encodes encoder_input and then decodes sequentially, with decoder_input as the starting input
    def translate(self, encoder_input, decoder_input):
        decoder_state = self.encoder(encoder_input)
        data_size = encoder_input.shape[0]
        decoder_output = np.zeros((data_size, self.target_pad_len - 1))
        for i in range(self.target_pad_len - 1):
            decoder_output_temp, decoder_state = self.decoder.predict([decoder_input, decoder_state], verbose = 0, batch_size = 128)
            next_words = np.argmax(decoder_output_temp, axis = -1)
            decoder_input = next_words
            decoder_output[:, i] = next_words.reshape((data_size,))
        return decoder_output
  
#Prevodi niz recenica
def translate_from_text(model, input_sentences, input_word_index, target_word_index, inverted_target_word_index, input_pad_len, target_pad_len):
    input_sentences = [input_sentence.replace('.', '').replace(',', '').replace('!','').replace('"','').replace('?','').lower() for input_sentence in input_sentences]
    data_length = len(input_sentences)
    target_placeholder_sentences = ['']*data_length
    encoder_input_data, _, _ = create_model_data(input_sentences, target_placeholder_sentences, input_word_index, target_word_index, input_pad_len, target_pad_len)
    decoder_input_data = np.zeros((data_length, 1)) + target_word_index['<Start>']
    decoder_output = model.translate(encoder_input_data, decoder_input_data)
    output_sentences = [' '.join(inverted_target_word_index[num] for num in output_sentence) for output_sentence in decoder_output]
    output_sentences = [output_sentence.split('<End>',1)[0].replace(' - ','-') for output_sentence in output_sentences]
    return output_sentences

#Evaluira model - Na osnovu prevoda i referenci vraca 5 razlicitih metrika za kvalitet prevoda: WER, smooth BLEU(1,2,3,4)
def evaluate(model, test_input_sentences, test_references, input_word_index, target_word_index, inverted_target_word_index, input_pad_len, target_pad_len):
    umlaut_dict = {'AE': 'Ä',
                    'OE': 'Ö',
                    'UE': 'Ü'}
    for key in umlaut_dict.keys():
        test_references = [text.replace(key, umlaut_dict[key]) for text in test_references]
    translations = translate_from_text(model, test_input_sentences, input_word_index, target_word_index, inverted_target_word_index, input_pad_len, target_pad_len)
    smooth_bleu4 = np.mean([sentence_bleu([reference.split()], translation.split(), smoothing_function = SmoothingFunction().method7) for reference, translation in zip(test_references, translations)])
    smooth_bleu3 = np.mean([sentence_bleu([reference.split()], translation.split(), smoothing_function = SmoothingFunction().method7, weights = [1/3, 1/3, 1/3]) for reference, translation in zip(test_references, translations)])
    smooth_bleu2 = np.mean([sentence_bleu([reference.split()], translation.split(), smoothing_function = SmoothingFunction().method7, weights = [1/2, 1/2]) for reference, translation in zip(test_references, translations)])
    smooth_bleu1 = np.mean([sentence_bleu([reference.split()], translation.split(), smoothing_function = SmoothingFunction().method7, weights = [1]) for reference, translation in zip(test_references, translations)])
    #Smoothing je preporucen za recenice, posebno kratke, a glossovane recenice prirodno imaju manji broj reci
    wer = np.mean([edit_distance(reference.split(), translation.split())/len(reference.split()) for reference, translation in zip(test_references, translations)])
    #metric_dict = {'wer': wer, 'smooth_bleu1': smooth_bleu1, 'smooth_bleu2':smooth_bleu2, 'smooth_bleu3': smooth_bleu3, 'smooth_bleu4': smooth_bleu4}
    return wer, smooth_bleu4, smooth_bleu3, smooth_bleu2, smooth_bleu1

#Trenira model na train_data i evaluira ga na val_data
#Embedding learning rate je poseban learning_rate koji se koristi u embedding slojevima, iz razloga sto oni vec imaju pretrenirane podatke za pocetne vrednosti
#Model se trenira dok val_loss ne krene da raste, i cuva tezine epohe koja ima najbolji val_loss
#Koristi se u cv_evaluate, napravio funkciju jer inace dolazi do prekoracenje GPU RAMa, neko je napisao da je do unakrsne validacije
def train_and_evaluate(train_data, val_data, epochs = 200, batch_size = 128, learning_rate = 0.001, latent_dim = 256, dropout_rate = 0.5, embedding_learning_rate = 0.001):
     
     input_texts, target_texts = clean_texts(train_data.iloc[:,1], train_data.iloc[:,0])
     input_word_index, target_word_index, max_input_seq_len, max_target_seq_len = analyse_texts(input_texts, target_texts)
     input_pad_len = 80
     target_pad_len = 60
     num_input_words = len(input_word_index) - 1
     num_target_words = len(target_word_index) - 1
     #print(num_input_words)
     inverted_input_word_index = {value: key for key,value in input_word_index.items()}
     inverted_target_word_index = {value: key for (key,value) in target_word_index.items()}
     #print(len(inverted_input_word_index))
     input_embedding_matrix, target_embedding_matrix = load_embedding_data_get_matrices(inverted_input_word_index, inverted_target_word_index)
     print('Embeddings loaded.')
     encoder_input_data, decoder_input_data, decoder_output_data = create_model_data(input_texts, target_texts, input_word_index, target_word_index, input_pad_len, target_pad_len)
     #print(input_embedding_matrix.shape)
     
     input_texts_val, target_texts_val = clean_texts(val_data.iloc[:,1], val_data.iloc[:,0])
     encoder_input_data_val, decoder_input_data_val, decoder_output_data_val = create_model_data(input_texts_val, target_texts_val, input_word_index, target_word_index, input_pad_len, target_pad_len)
     
     #print('Data preprocessed.')
     model_gru = GRU_Translation_Model(num_input_words, num_target_words, input_embedding_matrix, target_embedding_matrix, latent_dim = latent_dim, dropout_rate = dropout_rate)
     #print('Model loaded.')
     other_layers = model_gru.layers[0].layers + model_gru.layers[1].layers #Mora da se prilagodi za transformer
     embedding_layers = [other_layers.pop(2), other_layers.pop(-9)] #Paznja! Mora se prilagoditi svaki put kad se model menja

     optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers = [(Adam(learning_rate), other_layers), (Adam(embedding_learning_rate), embedding_layers)])
     model_gru.compile(optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
     #print('Model compiled.')
     history = model_gru.fit([encoder_input_data, decoder_input_data], decoder_output_data, validation_data = ([encoder_input_data_val, decoder_input_data_val], decoder_output_data_val), epochs = epochs, batch_size = batch_size, callbacks = [early_stopping], verbose = 1)
     #print('Model fit.')
     best_epoch = np.argmin(history.history['val_loss']) + 1
     
     #print('Best epoch: ', best_epoch)
     best_loss = np.min(history.history['val_loss'])
     #print('Best loss:', best_loss)
     #print(model_gru.evaluate([encoder_input_data_val, decoder_input_data_val], decoder_output_data_val))
     
     wer, smooth_bleu4, smooth_bleu3, smooth_bleu2, smooth_bleu1 = evaluate(model_gru, input_texts_val, target_texts_val, input_word_index, target_word_index, inverted_target_word_index, input_pad_len, target_pad_len)
     return best_epoch, best_loss, wer, smooth_bleu4, smooth_bleu3, smooth_bleu2, smooth_bleu1
 

#Trenira po model za svaki fold, racuna WER, smooth BLEU(1,2,3,4), kao i val_loss i broj epoha do konvergencije
#Vraca podatke iz svake instance modela, odnosno za svaki fold, da bi se dalje procesirale
def cv_evaluate(train_val_data = None, df_folds = None, folds = 5, epochs = 200, batch_size = 128, learning_rate = 0.001, latent_dim = 256, dropout_rate = 0.5, embedding_learning_rate = None):
    if embedding_learning_rate == None:
        embedding_learning_rate = learning_rate
    if df_folds == None:
        df_np = train_val_data.to_numpy()
        np.random.shuffle(df_np)
        total_size = df_np.shape[0]
        fold_size = total_size/folds
        df_folds = [df_np[int(i*fold_size):int((i+1)*fold_size),] for i in range(folds)]
    #input_word_embeddings, target_word_embeddings = load_embedding_data() #Doslo je do prekoracenje memorije
    losses = []
    best_epochs = []
    wers = []
    smooth_bleu1s = []
    smooth_bleu2s = []
    smooth_bleu3s = []
    smooth_bleu4s = []
    for i in range(folds):
        train_folds = [fold for j, fold in enumerate(df_folds) if j!=i]
        train_folds_pd = [pd.DataFrame(data = fold) for fold in train_folds]
        train_data = pd.concat(train_folds_pd)
        val_data = pd.DataFrame(df_folds[i])
        print('Current Latent Dim:', latent_dim)
        print('Current Dropout Rate: ', dropout_rate)
        print('Current Fold: {}/{}'.format(i+1, folds))
        print('Current Learning Rate: ', learning_rate)
        print('Current Learning Rate Multiplier: ', embedding_learning_rate/learning_rate)
        
        best_epoch, best_loss, wer, smooth_bleu4, smooth_bleu3, smooth_bleu2, smooth_bleu1 = train_and_evaluate(train_data, val_data, epochs = epochs, batch_size = batch_size, learning_rate = learning_rate, latent_dim = latent_dim, dropout_rate = dropout_rate, embedding_learning_rate = embedding_learning_rate)
        best_epochs.append(best_epoch)
        losses.append(best_loss)
        wers.append(wer)
        smooth_bleu4s.append(smooth_bleu4)
        smooth_bleu3s.append(smooth_bleu3)
        smooth_bleu2s.append(smooth_bleu2)
        smooth_bleu1s.append(smooth_bleu1)
    return best_epochs, losses, wers, smooth_bleu4s, smooth_bleu3s, smooth_bleu2s, smooth_bleu1s

#Evaluira modele za razlicite vrednosti latent_dim i dropout_rate
#U 3d matrici cuva rezultate, treca dimenzija predstavlja vrednosti za razlicite foldove, uprosecavanjem se dobija zeljena metrika
#Isto cuva i broj epoha do konvergencije 
def cv_grid_search(df, dropout_rates, latent_dims, epochs = 200, learning_rate = 0.0002, folds = 5):
    df_np = df.to_numpy()
    np.random.shuffle(df_np)
    total_size = df_np.shape[0]
    fold_size = total_size/folds
    df_folds = [df_np[int(i*fold_size):int((i+1)*fold_size),] for i in range(folds)]
    
    loss_matrix = np.zeros((len(latent_dims),len(dropout_rates), folds))
    epoch_matrix = np.zeros((len(latent_dims),len(dropout_rates), folds))
    wer_matrix = np.zeros((len(latent_dims),len(dropout_rates), folds))
    smooth_bleu4_matrix = np.zeros((len(latent_dims),len(dropout_rates), folds))
    smooth_bleu3_matrix = np.zeros((len(latent_dims),len(dropout_rates), folds))
    smooth_bleu2_matrix = np.zeros((len(latent_dims),len(dropout_rates), folds))
    smooth_bleu1_matrix = np.zeros((len(latent_dims),len(dropout_rates), folds))
    for i in range(len(latent_dims)):
        for j in range(len(dropout_rates)):
            best_epochs, losses, wers, smooth_bleu4s, smooth_bleu3s, smooth_bleu2s, smooth_bleu1s = cv_evaluate(df_folds = df_folds, folds = folds, epochs = epochs, learning_rate = learning_rate, latent_dim = latent_dims[i], dropout_rate = dropout_rates[j])
            print(losses)
            print(best_epochs)
            loss_matrix[i,j,:] = losses
            epoch_matrix[i,j,:] = best_epochs
            wer_matrix[i,j,:] = wers
            smooth_bleu4_matrix[i,j,:] = smooth_bleu4s
            smooth_bleu3_matrix[i,j,:] = smooth_bleu3s
            smooth_bleu2_matrix[i,j,:] = smooth_bleu2s
            smooth_bleu1_matrix[i,j,:] = smooth_bleu1s
    #Pakuju se rezultati u dictionary radi intuitivnijeg poziva funkcije
    metrics_dict = {'loss': loss_matrix, 'epoch': epoch_matrix, 'wer': wer_matrix, 'smooth_bleu4': smooth_bleu4_matrix, 'smooth_bleu3': smooth_bleu3_matrix, 'smooth_bleu2': smooth_bleu2_matrix, 'smooth_bleu1': smooth_bleu1_matrix }
    return metrics_dict

start_time = time.time()
    
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

#Hiperparametri za optimizaciju: dropout rate i latentna dimenzija
dropout_rates = [0.3, 0.5, 0.7]
latent_dims = [256, 512]
learning_rate = 0.0002
folds = 5

#dropout_rates = [0.1]
#latent_dims = [16]

#Izvrsava se grid search nad hiperparametrima, i radi se unakrsna validacija za evaluaciju performansi
#Metrika nad kojom se vrsi selekcija je smooth BLEU4

metrics = cv_grid_search(df_train_val, dropout_rates, latent_dims, epochs = 200, learning_rate = learning_rate, folds = folds)
average_bleu4 = np.mean(metrics['smooth_bleu4'], axis = -1)

plt.title('Average smooth BLEU4, crossvalidated')
plt.xlabel('Dropout Rate')
plt.xticks(range(len(dropout_rates)), dropout_rates)
plt.ylabel('Latent Dim')
plt.yticks(range(len(latent_dims)),latent_dims)
plt.imshow(average_bleu4)
plt.colorbar()
plt.show()

best_config_index = np.unravel_index(np.argmax(average_bleu4), average_bleu4.shape)
best_latent_dim = latent_dims[best_config_index[0]]
best_dropout_rate = dropout_rates[best_config_index[1]]

#Nakon sto se izaberu dropout rate i latent dim, optimizuje se unakrsnom validacijom i nad parametrom learning_rate_multiplier
#learning_rate_multiplier predstavlja odnos izmedju stope ucenja embedding slojeve i ostalih slojeva, koji je do sad imao podrazumevanu vrednost 1
#Ideja je da posto su embedding slojevi pretrenirani, mozda zahtevaju znacajno slabiji learning rate
#Razlog zasto ovo nije bilo deo grid searcha je vremenski, predugo bi trajalo da se odradi unakrsna validacija za 3*2*5 = 30 razlicitih modela

#learning_rate = 0.1
learning_rate_multipliers = [0.03, 0.1, 0.3, 1, 3]
#learning_rate_multipliers = [100]
best_epoch_array = []
bleu4_array = []
for i in range(len(learning_rate_multipliers)):
    df_np = df_train_val.to_numpy()
    np.random.shuffle(df_np)
    total_size = df_np.shape[0]
    fold_size = total_size/folds
    df_folds = [df_np[int(i*fold_size):int((i+1)*fold_size),] for i in range(folds)]
    best_epochs, _, _, smooth_bleu4s, _, _, _ = cv_evaluate(df_train_val, df_folds, folds = folds, epochs = 200, learning_rate = learning_rate, embedding_learning_rate = learning_rate_multipliers[i]*learning_rate, latent_dim = best_latent_dim, dropout_rate = best_dropout_rate)
    best_epoch_array.append(best_epochs)
    bleu4_array.append(smooth_bleu4s)

bleu4_array = np.array(bleu4_array)
best_epoch_array = np.array(best_epoch_array)
best_multiplier_index = np.argmax(np.mean(bleu4_array, -1))
bleu4_avg = np.mean(bleu4_array[best_multiplier_index])
print("Average BLEU4(smooth) on validation: ", bleu4_avg)
best_multiplier = learning_rate_multipliers[best_multiplier_index]
print("Best Multiplier for Embedding Learning Rates:", best_multiplier)
best_multiplier_epochs = best_epoch_array[best_multiplier_index]
print('Epochs to convergence on all folds: ', best_multiplier_epochs) #Gledamo koliko je epoha bilo potrebno do konvergencije
epoch_avg = np.mean(best_multiplier_epochs)#Prosek koristimo za broj epoha treniranja modela na trening i validacionom skupu


#Nakon sto su svi hiperparametri odabrani, vrsimo finalnu evaluaciju modela nad test skupom
#Model se trenira tako sto se odvoji mali deo podataka za evaluaciju tokom treniranja, i zavrsava se trening kada performanse nad ovim skupom krenu da opadaju

df_train_val_np = df_train_val.to_numpy()
np.random.shuffle(df_train_val_np)
split_size = 300
train_data = df_train_val_np[:train_size + val_size - split_size,]
val_data = df_train_val_np[train_size + val_size - split_size:,]
input_texts, target_texts = clean_texts(train_data[:,1], train_data[:,0])
input_word_index, target_word_index, max_input_seq_len, max_target_seq_len = analyse_texts(input_texts, target_texts)
input_pad_len = 80
target_pad_len = 60
num_input_words = len(input_word_index) - 1
num_target_words = len(target_word_index) - 1
inverted_input_word_index = {value: key for key,value in input_word_index.items()}
inverted_target_word_index = {value: key for (key,value) in target_word_index.items()}
input_embedding_matrix, target_embedding_matrix = load_embedding_data_get_matrices(inverted_input_word_index, inverted_target_word_index)
encoder_input_data, decoder_input_data, decoder_output_data = create_model_data(input_texts, target_texts, input_word_index, target_word_index, input_pad_len, target_pad_len)

input_texts_val, target_texts_val = clean_texts(val_data[:,1], val_data[:,0])
encoder_input_data_val, decoder_input_data_val, decoder_output_data_val = create_model_data(input_texts_val, target_texts_val, input_word_index, target_word_index, input_pad_len, target_pad_len)

input_texts_test, target_texts_test = clean_texts_df(df_test)
encoder_input_data_test, decoder_input_data_test, decoder_output_data_test = create_model_data(input_texts_test, target_texts_test, input_word_index, target_word_index, input_pad_len, target_pad_len)


model_for_evaluation = GRU_Translation_Model(num_input_words, num_target_words, input_embedding_matrix, target_embedding_matrix)

other_layers = model_for_evaluation.layers[0].layers + model_for_evaluation.layers[1].layers #Mora da se prilagodi za transformer
embedding_layers = [other_layers.pop(2), other_layers.pop(-9)] #Paznja! Mora se prilagoditi svaki put kad se model menja
optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers = [(Adam(learning_rate), other_layers), (Adam(best_multiplier*learning_rate), embedding_layers)])
model_for_evaluation.compile(optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['acc'])

#stojalo start_from_epoch = int(epoch_avg*0.7), ali iz nekog razloga ne prepoznaje argument
early_stopping_safe = EarlyStopping(patience = 20, restore_best_weights = True, monitor = 'val_loss', mode = 'min', verbose = 1)
history = model_for_evaluation.fit([encoder_input_data, decoder_input_data], decoder_output_data, validation_data = ([encoder_input_data_val, decoder_input_data_val], decoder_output_data_val), epochs = 200, batch_size = 128, verbose = 1, callbacks = [early_stopping_safe])
model_for_evaluation.summary()
epoch_counter = range(len(history.history['loss']))
fig, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(epoch_counter, history.history['loss'], label = 'Train Loss')
ax1.plot(epoch_counter, history.history['val_loss'], label = 'Validation Loss', linestyle = 'dashed')
ax1.legend()
ax2.plot(epoch_counter, history.history['acc'], label = 'Train accuracy')
ax2.plot(epoch_counter, history.history['val_acc'], label = 'Validation Accuracy', linestyle = 'dashed')
ax2.legend()

wer, smooth_bleu4, smooth_bleu3, smooth_bleu2, smooth_bleu1 = evaluate(model_for_evaluation, input_texts_test, target_texts_test, input_word_index, target_word_index, inverted_target_word_index, input_pad_len, target_pad_len)
print('Results on test data:')
print('Word Error Rate: ', wer) #76
print('BLEU4(smooth): ', smooth_bleu4) #11.7
print('BLEU3(smooth): ', smooth_bleu3) #17.1
print('BLEU2(smooth): ', smooth_bleu2) #25.2
print('BLEU1(smooth): ', smooth_bleu1) #36


end_time = time.time()
print('Total time: ', end_time - start_time)


# =============================================================================
# start_time = time.time()
# loss_matrix, epoch_matrix = cv_grid_search(df_train_val)
# end_time = time.time()
# print("Total Time", end_time - start_time)
# print(loss_matrix)
# print(epoch_matrix)
# =============================================================================


# =============================================================================
# #data_size = 1419
# #data_size = 2838
# #data_size = 4258
# #data_size = 5677
# #data_size = 6092
# data_size = 7096
# 
# input_texts, target_texts = clean_texts_df(df_train.iloc[:data_size,:])
# input_word_index, target_word_index, max_input_seq_len, max_target_seq_len = analyse_texts(input_texts, target_texts)
# input_pad_len = 80
# target_pad_len = 60
# num_input_words = len(input_word_index) - 1
# print(num_input_words)
# num_target_words = len(target_word_index) - 1
# inverted_input_word_index = {value: key for key,value in input_word_index.items()}
# inverted_target_word_index = {value: key for (key,value) in target_word_index.items()}
# input_embedding_matrix, target_embedding_matrix = load_embedding_data_get_matrices(inverted_input_word_index, inverted_target_word_index)
# encoder_input_data, decoder_input_data, decoder_output_data = create_model_data(input_texts, target_texts, input_word_index, target_word_index, input_pad_len, target_pad_len)
# 
# input_texts_val, target_texts_val = clean_texts_df(df_val)
# encoder_input_data_val, decoder_input_data_val, decoder_output_data_val = create_model_data(input_texts_val, target_texts_val, input_word_index, target_word_index, input_pad_len, target_pad_len)
# 
# print('Data preprocessed.')
# model_gru = GRU_Translation_Model(num_input_words, num_target_words, input_embedding_matrix, target_embedding_matrix)
# 
# model_gru.compile(optimizer = Adam(0.0002), loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
# 
# history = model_gru.fit([encoder_input_data, decoder_input_data], decoder_output_data, validation_data = ([encoder_input_data_val, decoder_input_data_val], decoder_output_data_val), epochs = 500, batch_size = 128, callbacks = [early_stopping], verbose = 1)
# epochs_to_converge_for_data_size = np.argmin(history.history['val_loss']) + 1
# print("Number of epochs to converge for data size {} is : {}".format(data_size, epochs_to_converge_for_data_size))
# =============================================================================
# =============================================================================
# epoch_counter = range(len(history.history['loss']))
# fig, (ax1, ax2) = plt.subplots(2,1)
# ax1.plot(epoch_counter, history.history['loss'], label = 'Train Loss')
# ax1.plot(epoch_counter, history.history['val_loss'], label = 'Validation Loss', linestyle = 'dashed')
# ax1.legend()
# ax2.plot(epoch_counter, history.history['acc'], label = 'Train accuracy')
# ax2.plot(epoch_counter, history.history['val_acc'], label = 'Validation Accuracy', linestyle = 'dashed')
# ax2.legend()
# =============================================================================








# =============================================================================
# other_layers = model_gru.layers[0].layers + model_gru.layers[1].layers
# embedding_indexes = [layer for layer in enumerate(other_layers) if 'embedding' in layer.name]
# embedding_layers = [other_layers.pop(2), other_layers.pop(-9)] #Paznja! Mora se prilagoditi svaki put kad se model menja
# 
# optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers = [(Adam(0.01), other_layers), (Adam(0.0002), embedding_layers)])
# =============================================================================
