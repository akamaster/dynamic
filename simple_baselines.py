import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import visualize_representation

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

from sklearn.cross_validation import train_test_split
data_train, data_test, labels_train, labels_test = train_test_split(data, labels,
                                                                    test_size=0.15,
                                                                    stratify=labels,
                                                                    random_state=42)
print(data_train.shape, data_test.shape)
data=None
labels=None

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.base import TransformerMixin
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support

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
    """
    Note: excecution of this method takes a lot of time: using ec2 c4.8xlarge, results yielded
    after 7 hours. Please refer to 'best_params.txt' to get optimal parameters for all experiments.
    :return:
    """
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

    grid_search = GridSearchCV(pipeline, param_grid=params, cv=StratifiedKFold(labels, 3, shuffle=True), n_jobs=-1)

    grid_search.fit(data_train, labels_train)
    print("Best params for bag of words:", file=log_file)
    print(grid_search.best_params_, file=log_file)
    print(grid_search.best_score_, file=log_file)
    print("Score is", file=log_file)


    pipeline.set_params(**grid_search.best_params_).set_params(forest__n_jobs=-1).fit(data_train,labels_train)
    print(pipeline.score(data_test, labels_test), file=log_file)
    print('Stats for Bag of Words: Rows - precision, recall, f1, support; '
          'Columns: environment active lifestyle physical capacity other', file=log_file)
    print(precision_recall_fscore_support(labels_test, pipeline.predict(data_test),
              labels=['environment', 'active lifestyle', 'physical capacity', 'other']), file=log_file)


def experiment2_tfidf():
    """
    Note: excecution of this method takes a lot of time: using ec2 c4.8xlarge, results yielded
    after 9 hours. Please refer to 'best_params.txt' to get optimal parameters for all experiments.
    :return:
    """
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

    grid_search = GridSearchCV(pipeline, param_grid=params, cv=StratifiedKFold(labels, 3, shuffle=True), n_jobs=-1)

    grid_search.fit(data_train, labels_train)
    print("Best params for tfidf:", file=log_file)
    print(grid_search.best_params_, file=log_file)
    print(grid_search.best_score_, file=log_file)
    print("Score is", file=log_file)

    pipeline.set_params(**grid_search.best_params_).set_params(forest__n_jobs=-1).fit(data_train, labels_train)
    print(pipeline.score(data_test, labels_test), file=log_file)
    print('Stats for TFIDF: Rows - precision, recall, f1, support; '
          'Columns: environment active lifestyle physical capacity other', file=log_file)

    print(precision_recall_fscore_support(labels_test, pipeline.predict(data_test),
              labels=['environment', 'active lifestyle', 'physical capacity', 'other']), file=log_file)


def visualize_experiments():
    pipeline_repr = Pipeline([('vectorizer', CountVectorizer(tokenizer=tokenize, max_features=2000)),
                         ('denser', DenseTransformer())])
    fig = visualize_representation(pipeline_repr.fit_transform(data_train),labels_train,
                             title='Visualization of feature space of Bag of Words representation')

    fig.savefig('bag_of_words_vis.png', dpi=400)

    pipeline = Pipeline([('vectorizer', TfidfVectorizer(tokenizer=tokenize, max_features=2000, smooth_idf=True)),
                         ('denser', DenseTransformer())
                         ])

    fig = visualize_representation(pipeline.fit_transform(data_train), labels_train,
                             title='Visualization of feature space of TfIdf representation')
    fig.savefig('tfidf_repr.png', dpi=400)


# Uncomment lines below to obtain best params.
experiment_1_bag_of_word()
experiment2_tfidf()

#visualize_experiments()
