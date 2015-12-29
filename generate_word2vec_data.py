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

from sklearn.cross_validation import train_test_split
data_train, data_test, labels_train, labels_test = train_test_split(data, labels,
                                                                    test_size=0.15,
                                                                    stratify=labels,
                                                                    random_state=42)
print(data_train.shape, data_test.shape)
data=None
labels=None



model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

def generate_for(data, labels):
    vec_words = []
    sequence_lengths = []
    for line in data:
        tokens = nltk.word_tokenize(line.lower())
        vec_words.append(list(model[t] for t in tokens if t not in string.punctuation and t in model))
        sequence_lengths.append(len(vec_words[-1]))

    number_of_examples = len(vec_words)
    len_of_seq = max(len(sentence) for sentence in vec_words)

    word2vec_repr = []
    word2vec_lens = []
    usable_index = np.array([True if len(sentence) > 0 else False for sentence in vec_words])
    word2vec_vectors_full = np.zeros((number_of_examples, len_of_seq, 300))
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
    return word2vec_repr, word2vec_labels, word2vec_vectors_full[usable_index,:,:], word2vec_lens


word2vec_repr_train, word2vec_labels_train, word2vec_vectors_full_train, word2vec_lens_train\
    = generate_for(data_train, labels_train)
word2vec_repr_test, word2vec_labels_test, word2vec_vectors_full_test, word2vec_lens_test \
    = generate_for(data_test, labels_test)


np.savez('word2vec_sum_repr.npz',
         data_train=word2vec_repr_train, labels_train=word2vec_labels_train,
         data_test =word2vec_repr_test,   labels_test=word2vec_labels_test)

print(word2vec_vectors_full_train.shape, word2vec_vectors_full_test.shape)

np.savez('word2vec_full_repr.npz',
         data_train=word2vec_vectors_full_train, labels_train=word2vec_labels_train, seq_lens_train=word2vec_lens_train,
         data_test =word2vec_vectors_full_test,   labels_test=word2vec_labels_test,  seq_lens_test=word2vec_lens_test)