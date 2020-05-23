import numpy as np
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Activation, Embedding
from tensorflow.keras.models import Model

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
	vocab_len = len(word_to_index) + 1
	emb_dim = word_to_vec_map["cucumber"].shape[0]
	emb_matrix = np.zeros((vocab_len, emb_dim))

	for word, idx in word_to_index.items():
		emb_matrix[idx, :] = word_to_vec_map[word]

	embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)

	embedding_layer.build((None,))
	embedding_layer.set_weights([emb_matrix])

	return embedding_layer

def define_model(input_shape, word_to_vec_map, word_to_index):
	sentence_indices = Input(shape= input_shape, dtype='int32')
	embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
	embeddings = embedding_layer(sentence_indices)
	X = LSTM(128, return_sequences=True)(embeddings)
	X = Dropout(rate=0.5)(X)
	X = LSTM(128, return_sequences=False)(X)
	X = Dropout(rate=0.5)(X)
	X = Dense(units=5, activation='softmax')(X)
	X = Activation('softmax')(X)

	model = Model(sentence_indices, X)

	return model