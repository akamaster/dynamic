import numpy as np
from utils import visualize_representation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

data = np.load('data.npz')['arr_0']
labels = data[:,1]
data = data[:,0]
tokens = [tokenize(line) for line in data]

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.base import TransformerMixin
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

log_file = open('logs.txt', mode='w')

class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, *args,**kwargs):
        return {}

def experiment_1_bag_of_word():
    pipeline_repr = Pipeline([('vectorizer', CountVectorizer(tokenizer=tokenize, max_features=1000)),
                         ('denser', DenseTransformer())])
    fig = visualize_representation(pipeline_repr.fit_transform(data),labels,
                             title='Visualization of feature space of Bag of Words representation')

    fig.savefig('bag_of_words_vis.png', dpi=400)

    pipeline = Pipeline([('vectorizer', CountVectorizer(tokenizer=tokenize, max_features=1000)),
                         ('denser', DenseTransformer()),
                         ('pca', PCA(n_components=0.8)),
                         ('forest', RandomForestClassifier(n_estimators=10000))])

    params = dict(pca__n_components=[0.9, 0.8, 0.7, 0.6],
              forest__n_estimators=[100, 200, 300, 600,900,1000,2000,4000],
              vectorizer__max_features=[100,500,1000,2000],
              forest__criterion=['gini', 'entropy'],
              forest__max_depth=[None,100, 50, 75],
              forest__min_samples_split=[2,4,8,10,100])

    grid_search = GridSearchCV(pipeline, param_grid=params, cv=StratifiedKFold(labels, 5, shuffle=True), n_jobs=-1)

    grid_search.fit(data, labels)
    print("Best params for bag of words:", file=log_file)
    print(grid_search.best_params_, file=log_file)
    print(grid_search.best_score_, file=log_file)
    print("Score is", file=log_file)
    scores = []

    skf = StratifiedKFold(labels, 5, random_state=42, shuffle=True)
    for train_id, test_id in skf:
        train_set = data[train_id]
        test_set = data[test_id]
        pipeline.set_params(**grid_search.best_params_).set_params(forest__n_jobs=-1).fit(train_set,labels[train_id])
        scores.append(pipeline.score(test_set, labels[test_id]))

    scores = np.array(scores)
    print(scores, file=log_file)
    print(scores.mean(), scores.std(), file=log_file)

def experiment2_tfidf():
    pipeline = Pipeline([('vectorizer', TfidfVectorizer(tokenizer=tokenize, max_features=1000, smooth_idf=True)),
                         ('denser', DenseTransformer())
                         ])

    fig = visualize_representation(pipeline.fit_transform(data), labels,
                             title='Visualiation of feature space of TfIdf representation')
    fig.savefig('tfidf_repr.png', dpi=400)

    pipeline = Pipeline([('vectorizer', TfidfVectorizer(tokenizer=tokenize, max_features=1000)),
                         ('denser', DenseTransformer()),
                         ('pca', PCA(n_components=0.8)),
                         ('forest', RandomForestClassifier(n_estimators=10000))])

    params = dict(pca__n_components=[0.9, 0.8, 0.7, 0.6],
              forest__n_estimators=[100, 200, 300, 600,900,1000,2000,4000],
              vectorizer__max_features=[100,500,1000,2000],
              vectorizer__smooth_idf = [True, False],
              forest__criterion = ['gini', 'entropy'],
              forest__max_depth = [None, 100, 50, 75],
              forest__min_samples_split = [2,4,8,10,100])

    grid_search = GridSearchCV(pipeline, param_grid=params, cv=StratifiedKFold(labels, 5, shuffle=True), n_jobs=-1)

    grid_search.fit(data, labels)
    print("Best params for tfidf:", file=log_file)
    print(grid_search.best_params_, file=log_file)
    print(grid_search.best_score_, file=log_file)
    print("Score is", file=log_file)
    scores = []

    skf = StratifiedKFold(labels, 5, random_state=42, shuffle=True)
    for train_id, test_id in skf:
        train_set = data[train_id]
        test_set = data[test_id]
        pipeline.set_params(**grid_search.best_params_).set_params(forest__n_jobs=-1).fit(train_set,labels[train_id])
        scores.append(pipeline.score(test_set, labels[test_id]))

    scores = np.array(scores)
    print(scores, file=log_file)
    print(scores.mean(), scores.std(), file=log_file)

experiment_1_bag_of_word()
experiment2_tfidf()
