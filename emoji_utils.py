import emoji
import numpy as np 
import pandas as pd

emoji_dictionary = {"0": "\u2764\uFE0F", 
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

def read_glove_vecs(glove_file):
	with open(glove_file, 'r', encoding='utf8') as f:
		words = set()
		word_to_vec_map = dict()
		for line in f:
			line = line.strip().split()
			curr_word = line[0]
			words.add(curr_word)
			word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
		i = 1
		words_to_index = dict()
		index_to_words = dict()

		for w in sorted(words):
			words_to_index[w] = i 
			index_to_words[i] = w
			i += 1
		return words_to_index, index_to_words, word_to_vec_map

def read_csv(filename):
	df = pd.read_csv(filename, header=None, usecols=[0,1])
	dataset = np.array(df)
	phrase, emoji = [], []

	for p, e in dataset:
		if '\t' in p:
			p = p[:-1]
		phrase.append(p)
		emoji.append(e)
	X = np.array(phrase)
	Y = np.array(emoji, dtype=int)
	return X,Y



def label_to_emoji(val):
	return emoji.emojize(emoji_dictionary[str(val)], use_aliases=True)

def convert_to_one_hot(Y,C):
	Y = np.eye(C)[Y.reshape(-1)]
	return Y