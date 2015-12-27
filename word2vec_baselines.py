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

data_train = npz_file['data_train']
data_test = npz_file['data_test']
labels_train = npz_file['labels_train']
labels_test = npz_file['labels_test']
log_file = open('logs_word2vec_baselines.txt', mode='w')

def experiment_3():
    """
    Note: excecution of this method takes a lot of time: using ec2 c4.8xlarge, results yielded
    after 1 hours. Please refer to 'best_params.txt' to get optimal parameters for all experiments.
    :return:
    """
    pipeline = Pipeline([('forest', RandomForestClassifier(n_estimators=100))])
    params = dict(
              forest__n_estimators=[100, 200, 300, 600, 900,1000,2000,4000],
              forest__criterion=['gini', 'entropy'],
              forest__max_depth=[None,100, 50, 75],
              forest__min_samples_split=[2,4,8,10,100])

    grid_search = GridSearchCV(pipeline, param_grid=params, cv=StratifiedKFold(labels_train,3, shuffle=True), n_jobs=-1)

    grid_search.fit(data_train, labels_train)
    print("Best params for word2vec_random forest:", file=log_file)
    print(grid_search.best_params_, file=log_file)
    print(grid_search.best_score_, file=log_file)
    print("Score is", file=log_file)


    pipeline.set_params(**grid_search.best_params_).set_params(forest__n_jobs=-1).fit(data_train, labels_train)
    print(pipeline.score(data_test, labels_test), file=log_file)

def experiment_4():
    """
    Note: excecution of this method takes a relatively small time: using ec2 c4.8xlarge, results yielded
    after 5 minutes. Please refer to 'best_params.txt' to get optimal parameters for all experiments.
    :return:
    """
    pipeline = Pipeline([('svm', SVC())])
    params = dict(
              svm__C=[0.1, 1, 5, 10, 50, 100],
              svm__kernel=['rbf', 'poly', 'linear'])

    grid_search = GridSearchCV(pipeline, param_grid=params, cv=StratifiedKFold(labels_train,3, shuffle=True), n_jobs=-1)

    grid_search.fit(data_train, labels_train)
    print("Best params for word2vec_svm forest:", file=log_file)
    print(grid_search.best_params_, file=log_file)
    print(grid_search.best_score_, file=log_file)
    print("Score is", file=log_file)

    pipeline.set_params(**grid_search.best_params_).fit(data_train, labels_train)

    print(pipeline.score(data_test, labels_test), file=log_file)

def statistics():
    pass

experiment_3()
experiment_4()
log_file.close()

#TODO: generate basic statistics