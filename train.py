import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import pandas as pd
import numpy as np 
from utils import read_csv,read_glove_vecs,sentences_to_indices,convert_to_one_hot
from buildModel import define_model, pretrained_embedding_layer

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')

X_train, Y_train = read_csv('train_emoji.csv')
X_test, Y_test = read_csv('test.csv')

maxLen = len(max(X_train, key=len).split())

model = define_model((maxLen,), word_to_vec_map, word_to_index)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_one_hot = convert_to_one_hot(Y_train, C = 5)


print("===" * 20)
print("MODEL TRAINING STARTED")
history = model.fit(X_train_indices, Y_train_one_hot, epochs = 50, batch_size=32, shuffle=True)
print("MODEL TRAINING FINISHED")
print("===" * 20)
print("SAVING THE MODEL")
model_filename = 'emojify.h5'
model.save(model_filename)
print("===" * 20)
print("MODEL SAVED AS: ", model_filename)
print("===" * 20)
