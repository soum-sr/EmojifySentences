import pandas as pd
import numpy as np 
from emoji_utils import read_csv,read_glove_vecs
from buildModel import define_model, pretrained_embedding_layer

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')

X_train, Y_train = read_csv('train_emoji.csv')
X_test, Y_test = read_csv('tesss.csv')

maxLen = len(max(X_train, key=len).split())

model = define_model((maxLen,), word_to_vec_map, word_to_index)

