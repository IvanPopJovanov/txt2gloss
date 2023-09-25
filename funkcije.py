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

embedding_size = 300

class CustomDropout(Layer):
    """
    Dropout koji umesto nule, menja inpute sa zadatom vrednoscu
    Takodje ne reskalira ostale inpute
    Primena je na tekstualnom inputu, da se reci nasumicno zamene sa <Unknown> tokenom, radi boljeg procesiranja nepoznatih reci
    """
    def __init__(self, custom_value, rate, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.custom_value = custom_value
        self.rate = rate

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()
    
        if training:
            return tf.where(tf.logical_and(tf.random.uniform(shape=tf.shape(inputs)) < self.rate, inputs != 0), self.custom_value, inputs)
        return inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "custom_value": self.custom_value,
            "rate": self.rate
        })
        return config
    
def clean_texts(input_texts, target_texts):
    """
    Cisti zadate input i target tekstove
    Dodaje <Start> i <End> tokene
    Pravilno upisuje posebna Nemacka slova
    Takodje za Glossove crtu odvaja od reci koje spaja, da bi se tretirala kao poseban token
    """
    target_texts = ['<Start> ' + text + ' <End>' for text in target_texts]
    target_texts = [text.replace('-', ' - ') for text in target_texts]
    umlaut_dict = {'AE': 'Ä',
                   'OE': 'Ö',
                   'UE': 'Ü',
                   'AÜ': 'AUE'}
    for key in umlaut_dict.keys():
        target_texts = [text.replace(key, umlaut_dict[key]) for text in target_texts]
        
    return [input_texts, target_texts]

def clean_texts_df(data_frame):
    """
    Wrapper za clean_texts koji se direktno da dataframe primenjuje
    """
    input_texts = data_frame['translation']
    target_texts = data_frame['orth']
    return clean_texts(input_texts, target_texts)

def analyse_texts(input_texts, target_texts):
    """
    Prolazi kroz tekstove i izdvaja iz njih recnik. Reci koje se pojavljuju samo jednom se ignorisu, kasnije se mapiraju na poseban <Unknown> token
    Takodje belezi maksimalne duzine recenica za input i target
    """
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

def load_embedding_data():
    """
    Ucitava pretrenirane glove embedding vektore
    Potencijalno vrati samo jedan, s obzirom da su prakticno kopije, mozda bude bitno za memoriju
    """
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



def get_embedding_matrices(inverted_input_word_index, inverted_target_word_index, input_word_embeddings, target_word_embeddings):
    """
    Mapira reci iz recnika na odgovarajuce pretrenirane glove embeddinge
    Reci koje nemaju odgovarajuci embedding dobijaju nula vektor kao embedding
    """
    num_input_words = len(inverted_input_word_index) - 1
    num_target_words = len(inverted_target_word_index) - 1
    
    input_embedding_matrix = np.zeros((num_input_words + 1, embedding_size))
    for i in range(num_input_words):
        input_embedding_matrix[i + 1] = input_word_embeddings.get(inverted_input_word_index[i+1], np.zeros(embedding_size))

    target_embedding_matrix = np.zeros((num_target_words + 1, embedding_size))
    for i in range(num_target_words):
        target_embedding_matrix[i + 1] = target_word_embeddings.get(inverted_target_word_index[i+1], np.zeros(embedding_size))
    
    return [input_embedding_matrix, target_embedding_matrix]

def load_embedding_data_get_matrices(inverted_input_word_index, inverted_target_word_index):
    """
    Ucitava pretrenirane embedding vektora i mapira reci iz recnika na njih
    U sustini wrapper za load_embedding_data i get_embedding_matrices
    Prednost je sto ce ovako odmah nakon zavrsetka funkcije da se oslobodi memorije za sve embeddinge, koji zauzimaju oko 6GB
    """
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

def create_model_data(input_texts, target_texts, input_word_index, target_word_index, input_pad_len, target_pad_len):
    """
    Iz recenica kreira podatke koji su spremni da se unose u model
    """
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

def translate_from_text(model, input_sentences, input_word_index, target_word_index, inverted_target_word_index, input_pad_len, target_pad_len):
    """
    Prevodi niz recenica
    """
    input_sentences = [input_sentence.replace('.', '').replace(',', '').replace('!','').replace('"','').replace('?','').lower() for input_sentence in input_sentences]
    data_length = len(input_sentences)
    target_placeholder_sentences = ['']*data_length
    encoder_input_data, _, _ = create_model_data(input_sentences, target_placeholder_sentences, input_word_index, target_word_index, input_pad_len, target_pad_len)
    decoder_input_data = np.zeros((data_length, 1)) + target_word_index['<Start>']
    decoder_output = model.translate(encoder_input_data, decoder_input_data)
    output_sentences = [' '.join(inverted_target_word_index[num] for num in output_sentence) for output_sentence in decoder_output]
    output_sentences = [output_sentence.split('<End>',1)[0].replace(' - ','-') for output_sentence in output_sentences]
    return output_sentences

def evaluate(model, test_input_sentences, test_references, input_word_index, target_word_index, inverted_target_word_index, input_pad_len, target_pad_len):
    """
    Evaluira model - Na osnovu prevoda i referenci vraca 5 razlicitih metrika za kvalitet prevoda: WER, smooth BLEU(1,2,3,4)
    """
    umlaut_dict = {'AE': 'Ä',
                   'OE': 'Ö',
                   'UE': 'Ü',
                   'AÜ': 'AUE'}
    for key in umlaut_dict.keys():
        test_references = [text.replace(key, umlaut_dict[key]) for text in test_references]
    test_references = [text.replace('<Start>', '') for text in test_references]
    test_references = [text.replace('<End>', '') for text in test_references]
    test_references = [text.replace(' - ', '-') for text in test_references]
    translations = translate_from_text(model, test_input_sentences, input_word_index, target_word_index, inverted_target_word_index, input_pad_len, target_pad_len)
    smooth_bleu4 = np.mean([sentence_bleu([reference.split()], translation.split(), smoothing_function = SmoothingFunction().method7) for reference, translation in zip(test_references, translations)])
    smooth_bleu3 = np.mean([sentence_bleu([reference.split()], translation.split(), smoothing_function = SmoothingFunction().method7, weights = [1/3, 1/3, 1/3]) for reference, translation in zip(test_references, translations)])
    smooth_bleu2 = np.mean([sentence_bleu([reference.split()], translation.split(), smoothing_function = SmoothingFunction().method7, weights = [1/2, 1/2]) for reference, translation in zip(test_references, translations)])
    smooth_bleu1 = np.mean([sentence_bleu([reference.split()], translation.split(), smoothing_function = SmoothingFunction().method7, weights = [1]) for reference, translation in zip(test_references, translations)])
    #Smoothing je preporucen za recenice, posebno kratke, a glossovane recenice prirodno imaju manji broj reci
    wer = np.mean([edit_distance(reference.split(), translation.split())/len(reference.split()) for reference, translation in zip(test_references, translations)])
    return wer, smooth_bleu4, smooth_bleu3, smooth_bleu2, smooth_bleu1