from utils import visualize_representation
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC
import numpy as np

npz_file = np.load('word2vec_sum_repr.npz')

data = npz_file['data']
print(data.dtype)
labels = npz_file['labels']
print(np.isfinite(data).all(), data.shape, labels.shape)
log_file = open('logs_word2vec_baselines.txt', mode='w')

def experiment_3():
    """
    Note: excecution of this method takes a lot of time: using ec2 c4.8xlarge, results yielded
    after 7 hours. Please refer to 'best_params.txt' to get optimal parameters for all experiments.
    :return:
    """
    pipeline = Pipeline([('forest', RandomForestClassifier(n_estimators=100))])
    params = dict(
              forest__n_estimators=[100, 200, 300, 600, 900,1000,2000,4000],
              forest__criterion=['gini', 'entropy'],
              forest__max_depth=[None,100, 50, 75],
              forest__min_samples_split=[2,4,8,10,100])

    grid_search = GridSearchCV(pipeline, param_grid=params, cv=StratifiedKFold(labels,3, shuffle=True), n_jobs=-1)

    grid_search.fit(data, labels)
    print("Best params for word2vec_random forest:", file=log_file)
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

def experiment_4():
    """
    Note: excecution of this method takes a lot of time: using ec2 c4.8xlarge, results yielded
    after 7 hours. Please refer to 'best_params.txt' to get optimal parameters for all experiments.
    :return:
    """
    pipeline = Pipeline([('svm', SVC())])
    params = dict(
              svm__C=[0.1, 1, 5, 10, 50, 100],
              svm__kernel=['rbf', 'poly', 'linear'])

    grid_search = GridSearchCV(pipeline, param_grid=params, cv=StratifiedKFold(labels,3, shuffle=True), n_jobs=-1)

    grid_search.fit(data, labels)
    print("Best params for word2vec_random forest:", file=log_file)
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

experiment_3()
experiment_4()
log_file.close()