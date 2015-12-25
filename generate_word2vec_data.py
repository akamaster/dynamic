from gensim.models.word2vec import Word2Vec
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
import string


stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


data = np.load('data.npz')['arr_0']
labels = data[:,1]
data = data[:,0]

vec_words = []
model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
sequence_lengths = []
for line in data:
    tokens = nltk.word_tokenize(line.lower())
    vec_words.append(list(model[t] for t in tokens if t not in string.punctuation and t in model))
    sequence_lengths.append(len(vec_words[-1]))

number_of_examples = len(vec_words)
len_of_seq = max(len(sentence) for sentence in vec_words)
word2vec_vectors_full = np.zeros((number_of_examples, len_of_seq, 300))
word2vec_repr = []
word2vec_lens = []
usable_index = np.array([True if len(sentence) > 0 else False for sentence in vec_words])
print(sum(usable_index))
for i, sentence in enumerate(vec_words):
    len_ = len(sentence)
    if len(sentence) > 0:
        sentence = np.array(sentence)
        word2vec_repr.append(sentence.mean(axis=0))
        word2vec_vectors_full[i,:len_,:] = sentence
        word2vec_lens.append(len_)

word2vec_repr = np.array(word2vec_repr)
word2vec_labels = labels[usable_index]
print(word2vec_labels.shape, word2vec_repr.shape)

np.savez('word2vec_sum_repr.npz', data=word2vec_repr, labels=word2vec_labels)
np.savez('word2vec_full_repr.npz', data=word2vec_vectors_full, labels=word2vec_labels, seq_lens = word2vec_lens)