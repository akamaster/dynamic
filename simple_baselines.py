import numpy as np
from utils import visualize_representation
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

from sklearn.feature_extraction.text import CountVectorizer
#visualize_representation(x, data[:,1])
#plt.show()

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.base import TransformerMixin
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

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

pipeline = Pipeline([('vectorizer', CountVectorizer(tokenizer=tokenize, max_features=1000)),
                     ('denser', DenseTransformer()),
                     ('pca', PCA(n_components=0.8)),
                     ('forest', RandomForestClassifier(n_estimators=10000))])



params = dict(pca__n_components=[0.9, 0.8, 0.7],
              forest__n_estimators=[100, 200, 300],
              #forest__n_jobs=[-1]
              )
grid_search = GridSearchCV(pipeline, param_grid=params, cv=StratifiedKFold(labels, 2, shuffle=True), n_jobs=2)
grid_search.fit(data, labels)
print(grid_search.grid_scores_)
print(grid_search.best_params_)
print(grid_search.best_score_)


scores = []

skf = StratifiedKFold(labels, 5, random_state=42, shuffle=True)
for train_id, test_id in skf:
    train_set = data[train_id]
    label_train = labels[train_id]
    test_set = data[test_id]
    pipeline.set_params(**grid_search.best_params_).fit(train_set,labels[train_id])
    scores.append(pipeline.score(test_set, labels[test_id]))

scores = np.array(scores)
print(scores)
print(scores.mean(), scores.std())
