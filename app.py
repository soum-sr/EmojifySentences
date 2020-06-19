import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from flask import Flask
from flask import render_template, request

from buildModel import define_model, pretrained_embedding_layer
from utils import read_glove_vecs, sentences_to_indices, label_to_emoji, preprocess

app = Flask(__name__)

maxLen = 10

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')

model = define_model((maxLen,), word_to_vec_map, word_to_index)
model.load_weights('emojify.h5')


def emojify(sentences):
	sentences = sentences.split('.')

	out = []
	for sentence in sentences:
		s = preprocess(sentence)
		print("preprocessed_sentence: ", s)
		if len(s) != 0:
			s_arr = np.array([s])
			s_indices = sentences_to_indices(s_arr, word_to_index, maxLen)
			out.append(sentence+' ' + label_to_emoji(np.argmax(model.predict(s_indices))))
	out = '.'.join(out) + '.'
	return out



@app.route('/', methods=['GET', 'POST'])
def index():
	return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True)