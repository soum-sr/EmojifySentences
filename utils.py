import emoji
import numpy as np 
import pandas as pd
import string

emoji_dictionary = {"0": "\u2764\uFE0F", 
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

def preprocess(sentence):
	words = sentence.split()
	table = str.maketrans('','',string.punctuation)
	words = [w.translate(table) for w in words]
	words = [word for word in words if word.isalpha()]
	return ' '.join(words)
	




def read_glove_vecs(glove_file):
	with open(glove_file, 'r', encoding='utf8') as f:
		words = set()
		word_to_vec_map = {}
		for line in f:
			line = line.strip().split()
			curr_word = line[0]
			words.add(curr_word)
			word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
		i = 1
		words_to_index = {}
		index_to_words = {}
		for w in sorted(words):
			words_to_index[w] = i
			index_to_words[i] = w
			i = i + 1
	return words_to_index, index_to_words, word_to_vec_map

def read_csv(filename):
	df = pd.read_csv(filename, header=None, usecols=[0,1])
	dataset= np.array(df)
	phrase, emoji = list(), list()
	for p, e in dataset:
		if '\t' in p:
			p = p[:-1]
		phrase.append(p)
		emoji.append(e)
	X = np.array(phrase)
	y = np.array(emoji, dtype=int)
	return X,y

def sentences_to_indices(X, word_to_index, max_len):
	m = X.shape[0]
	X_indices = np.zeros((m, max_len))

	for i in range(m):
		sentence_words = X[i].lower().split()
		j = 0
		for w in sentence_words:
			X_indices[i,j] = word_to_index[w]
			j+=1
	return X_indices

def label_to_emoji(val):
    return emoji.emojize(emoji_dictionary[str(val)], use_aliases=True)

def convert_to_one_hot(Y,C):
	Y = np.eye(C)[Y.reshape(-1)]
	return Y

