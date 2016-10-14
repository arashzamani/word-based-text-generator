'''Example script to generate text using keras and word2vec

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from nltk import tokenize
import numpy as np
import random
import sys
import os
import nltk

import gensim, logging
import os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# a memory-friendly iterator
class MySentences(object):
    def __init__(self, dirname, min_word_count_in_sentence = 1):
        self.dirname = dirname
        self.min_word_count_in_sentence = min_word_count_in_sentence;
    
    def process_line(self, line):
        words = line.split()
        return words

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                processed_line = self.process_line(line)
                if (len(processed_line) >= self.min_word_count_in_sentence):
                    yield processed_line
                else:
                    continue

def generate_word2vec_train_files(input_dir, output_dir, sentence_start_token, sentence_end_token, unkown_token, word_min_count, word2vec_size):
    print('generate_word2vec_train_files...')
    tmp_word2vec_model = gensim.models.Word2Vec(min_count = word_min_count, size = word2vec_size)
    original_sentences = MySentences(input_dir)
    tmp_word2vec_model.build_vocab(original_sentences)
    original_word2vec_vocab = tmp_word2vec_model.vocab

    make_dir_if_not_exist(output_dir)
    for fname in os.listdir(input_dir):
        output_file = open(os.path.join(output_dir, fname), 'w')
        line_count = 0
        for line in open(os.path.join(input_dir, fname)):
            line = line.strip(' -=:\"\'_*\n')
            if len(line) == 0:
                continue
            sentences = tokenize.sent_tokenize(line)
            for idx, sentence in enumerate(sentences):
                words = sentence.split()
                for word_idx, word in enumerate(words):
                    if word not in original_word2vec_vocab:
                        words[word_idx] = unkown_token#TODO
                sentence = " ".join(word for word in words)
                sentences[idx] = sentence_start_token + ' ' + sentence + ' ' + sentence_end_token + '\n'
            line_count += len(sentences)
            output_file.writelines(sentences)
        output_file.close()
        print("line_count", line_count)

def train_word2vec_model(dataset_dir, save_model_file, word_min_count, word2vec_size):
    print('train_word2vec_model...')
    word2vec_model = gensim.models.Word2Vec(min_count = word_min_count, size = word2vec_size)
    train_sentences = MySentences(dataset_dir)
    word2vec_model.build_vocab(train_sentences)
    sentences = MySentences(dataset_dir)
    word2vec_model.train(sentences)
    word2vec_model.save(save_model_file)
    return word2vec_model

def load_existing_word2vec_model(model_file_path):
    model =None
    if os.path.exists(model_file_path):
        print("load existing model...")
        model = gensim.models.Word2Vec.load(model_file_path)
    return model

def generate_rnn_train_files(input_dir, output_dir, fixed_sentence_len, unkown_token, sentence_start_token, sentence_end_token):
    print('generate_rnn_train_files...')
    make_dir_if_not_exist(output_dir)

    long_than_fixed_len_count = 0;
    total_sentence_count = 0;
    for fname in os.listdir(input_dir):
        output_file = open(os.path.join(output_dir, fname), 'w')
        for sentence in open(os.path.join(input_dir, fname)):
            sentence = sentence.strip('\n')
            total_sentence_count += 1
            words = sentence.split()
            len_of_sentence = len(words)
            if len_of_sentence > fixed_sentence_len:
                long_than_fixed_len_count += 1
                continue
            elif len_of_sentence < fixed_sentence_len:
                for i in range(0, fixed_sentence_len - len_of_sentence):
                    sentence = sentence + ' ' + sentence_end_token
            output_file.write(sentence + '\n')
        output_file.close()
    print ("sentence longer than fixed_len : %d / %d" %(long_than_fixed_len_count, total_sentence_count))

def train_rnn_model(dataset_dir, fixed_sentence_len, word2vec_size, word2vec_model):
    # build the model: a single LSTM
    print('Build RNN model...')
    rnn_model = Sequential()
    rnn_model.add(LSTM(128, input_shape=(fixed_sentence_len, word2vec_size)))
    rnn_model.add(Dense(word2vec_size))
    rnn_model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    rnn_model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    
    print('Generating RNN train data...')
    X = [] #np.zeros((0, fixed_sentence_len, word2vec_size), dtype=np.float32)
    y = [] #np.zeros((0, word2vec_size), dtype=np.float32)
    sentences = MySentences(dataset_dir)
    for sentence in sentences:
        tmp_x = np.asarray([word2vec_model[w] for w in sentence[:-1]])
        tmp_y = np.asarray([word2vec_model[w] for w in sentence[1:]])
        tmp_x = np.zeros((fixed_sentence_len, word2vec_size), dtype=np.float32)
        for idx, word in enumerate(sentence):
            tmp_x[idx] = word2vec_model[word]
            X.append()
    # X, y = generate_rnn_train_data()
    print(X)
    print(y)
    print('Generate RNN train data end!')

    # rnn_model.fit()
    print('Build RNN model over!')

    return rnn_model

class Config:
    WORD2VEC_MODE_FILE = "./word2vec_model.model"
    ORIGINAL_TRAIN_DATASET_DIR = "./small_train_text"
    WORD2VEC_TRAIN_DATASET_DIR = "./small_word2vec_train_text"
    RNN_TRAIN_DATASET_DIR = "./small_rnn_train_text"
    SENTENCE_START_TOKEN = "SENTENCE_START_TOKEN"
    SENTENCE_END_TOKEN = "SENTENCE_END_TOKEN"
    UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
    FIXED_SENTENCE_LEN = 30
    MIN_COUNT = 2;
    WORD2VEC_SIZE = 20;

def make_dir_if_not_exist(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

def main():

    # word2vec train
    word2vec_model = load_existing_word2vec_model(Config.WORD2VEC_MODE_FILE)

    if word2vec_model == None:
        generate_word2vec_train_files(
            Config.ORIGINAL_TRAIN_DATASET_DIR, Config.WORD2VEC_TRAIN_DATASET_DIR,
            Config.SENTENCE_START_TOKEN, Config.SENTENCE_END_TOKEN, Config.UNKNOWN_TOKEN, Config.MIN_COUNT, Config.WORD2VEC_SIZE)

        word2vec_model = train_word2vec_model(Config.WORD2VEC_TRAIN_DATASET_DIR, Config.WORD2VEC_MODE_FILE, Config.MIN_COUNT, Config.WORD2VEC_SIZE)

    # rnn train
    generate_rnn_train_files(
        Config.WORD2VEC_TRAIN_DATASET_DIR, Config.RNN_TRAIN_DATASET_DIR,
        Config.FIXED_SENTENCE_LEN, Config.UNKNOWN_TOKEN,
        Config.SENTENCE_START_TOKEN, Config.SENTENCE_END_TOKEN)

    rnn_model = train_rnn_model(Config.RNN_TRAIN_DATASET_DIR, Config.FIXED_SENTENCE_LEN, Config.WORD2VEC_SIZE, word2vec_model)

main()


# if __name__ == "__main__":
#     main()

# 