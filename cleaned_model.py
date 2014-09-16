"""
Code: https://github.com/log0/higgs_boson
Author: Eric Chio "log0" <ckieric [dot] gmail [dot] com>
Date: 2014/09/15

Description:
Rank 23 solution to the Kaggle Higgs Boson Machine Learning Challenge.
Extensive documentation is inline.

Competition: https://www.kaggle.com/c/higgs-boson
Copyrights 2014, Eric Chio.
BSD license, 3 clauses.
"""

import csv
import math
import os
import random

import numpy as np
from sklearn.cross_validation import *
from sklearn.decomposition import *
from sklearn.ensemble import *
from sklearn.feature_selection import *
from sklearn.grid_search import *
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.pipeline import *
from sklearn.svm import *
from sklearn.tree import *

#########################################################################
# Metrics 
#########################################################################

def ams(s, b):
    return math.sqrt(2 * ((s + b + 10) * math.log(1.0 + s/(b + 10)) - s))

def get_ams_score(W, Y, Y_pred):
    s = W * (Y == 1) * (Y_pred == 1)
    b = W * (Y == 0) * (Y_pred == 1)
    s = np.sum(s)
    b = np.sum(b)
    return ams(s, b)

#########################################################################
# Models 
#########################################################################

def single_model():
    # Personal note: I've always been impressed by why some people got the
    # magical numbers in a model. Do you? I'll explain how I arrive with
    # the numbers below.
    # 
    # All of the parameters are determined through gridsearch (manually
    # and automated).
    # 
    # AdaBoostClassifer:
    #   I picked this model because:
    #     1) it supports weighted examples in Scikit
    #        library.
    #     2) it is good at telling the model to focus on the wronged examples.
    #
    #   Parameters:
    #   n_estimators: 20. Gridsearched value. I tried 15, 20, 25, etc. 20
    #     looks optimal
    #   learning_rate: 0.75. Gridsearched value. I tried 1.0, 0.9, 0.8, etc.
    #     I don't think a 0.7 and 0.6 mattered much.
    #
    # ExtraTreesClassifier:
    #   I picked ExtraTreesClassifier because since there are many examples,
    #   it benefits from subsampling. GradientBoostingClassifier is too slow
    #   and RandomForestClassifier does not take advantage of the subsampling.
    #   Subsampling is good when training data is abundant.
    #
    #   Parameters:
    #     n_estimators: 400. Gridsearched. More trees to an extend is usually
    #       good.
    #     max_features: 30. Gridsearched. Using more features is usually a good
    #       idea, noting that using fewer features increases the randomness,
    #       which makes the model good. If you use too few features for each
    #       tree, there will not be enough predictive power in each tree.
    #     max_depth: 12. Gridsearched. This can overfit really easy.
    #     min_samples_leaf: 100. Gridsearched. I tried tuning this up because
    #       we have many data instances to learn from. Turns out this helps.
    #     min_samples_split: 100. Gridsearched. I tried tuning this up because
    #       we have many data instances to learn from. Turns out this helps.
    classifier = AdaBoostClassifier(
            n_estimators = 20,
            learning_rate = 0.75,
            base_estimator = ExtraTreesClassifier(
                n_estimators = 400,
                max_features = 30,
                max_depth = 12,
                min_samples_leaf = 100,
                min_samples_split = 100,
                verbose = 1,
                n_jobs = -1))

    return classifier

#########################################################################
# Feature preprocessing
#########################################################################

def preprocess(X, X_test):
    # Impute missing data. This may not make actual sense, because some
    # data is missing because it really should be missing. However, scikit
    # models do not play well with missing data.
    imputer = Imputer(missing_values = -999.0, strategy = 'most_frequent')
    X = imputer.fit_transform(X)
    X_test = imputer.transform(X_test)

    # Create inverse log values of features which is positive in value.
    inv_log_cols = (0,1,2,3,4,5,7,8,9,10,12,13,16,19,21,23,26)
    X_inv_log_cols = np.log(1 / (1 + X[:, inv_log_cols]))
    X = np.hstack((X, X_inv_log_cols))
    X_test_inv_log_cols = np.log(1 / (1 + X_test[:, inv_log_cols]))
    X_test = np.hstack((X_test, X_test_inv_log_cols))

    # Scaling usually helps a machine learning algorithm. Tree-based models
    # should not be of much help, but doesn't hurt.
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    return X, X_test

#########################################################################
# Training run
#########################################################################

def train_and_predict(X, W, Y, X_test):
    # Preprocess the features.
    X, X_test = preprocess(X, X_test)

    classifier = single_model()

    # Use the weight to hint the model to not treat each data instance as
    # equal, but put more attention to some of the data instance with more
    # weight.
    classifier.fit(X, Y, sample_weight = W)

    Y_pred = classifier.predict_proba(X)[:,1]
    Y_test_pred = classifier.predict_proba(X_test)[:,1]

    # We only predict an instance as signal if we are really confident,
    # because AMS score penalizses false positives far more than false
    # negatives.
    signal_threshold = 83
    cut = np.percentile(Y_test_pred, signal_threshold)
    thresholded_Y_pred = Y_pred > cut
    thresholded_Y_test_pred = Y_test_pred > cut
    
    return [Y_test_pred, thresholded_Y_test_pred]

#########################################################################
# Submission generation
#########################################################################

def write_submission_file(ids_test, Y_test_pred, thresholded_Y_test_pred):
    ids_probs = np.transpose(np.vstack((ids_test, Y_test_pred)))
    ids_probs = np.array(sorted(ids_probs, key = lambda x: -x[1]))
    ids_probs_ranks = np.hstack((
        ids_probs,
        np.arange(1, ids_probs.shape[0]+1).reshape((ids_probs.shape[0], 1))))

    test_ids_map = {}
    for test_id, prob, rank in ids_probs_ranks:
        test_id = int(test_id)
        rank = int(rank)
        test_ids_map[test_id] = rank

    f = open('submission.%s.out' % __file__, 'wb')
    writer = csv.writer(f)
    writer.writerow(['EventId', 'RankOrder', 'Class'])
    for i, pred in enumerate(thresholded_Y_test_pred):
        event_id = int(ids_test[i])
        rank = test_ids_map[ids_test[i]]
        klass = pred and 's' or 'b'
        writer.writerow([event_id, rank, klass])
    f.close()

if __name__ == '__main__':
    # Fix CPU affinity caused by Numpy.
    # http://stackoverflow.com/questions/15639779/what-determines-whether-different-python-processes-are-assigned-to-the-same-or-d
    os.system('taskset -p 0xffffffff %d' % os.getpid())

    # It is important to pick the right seed! Reduce randomness wherever
    # possible. Especially in a CV loop, so your solutions are more
    # comparable.
    seed = 512
    random.seed(seed)

    # Load training data. Point this to your training data.
    print 'Loading training data.'
    data = np.loadtxt('../data/training.csv', \
            delimiter=',', \
            skiprows=1, \
            converters={32: lambda x:int(x=='s'.encode('utf-8'))})

    X = data[:,1:31]
    Y = data[:,32]
    W = data[:,31]

    # Work on test data and generate submission. Point this to your
    # testing data.
    print 'Loading testing data.'
    test_data = np.loadtxt('../data/test.csv', \
        delimiter=',', \
        skiprows=1)

    ids_test = test_data[:,0]
    X_test = test_data[:,1:31]
    W = data[:,31]

    # Train model now. 
    Y_test_pred, thresholded_Y_test_pred = train_and_predict(X, W, Y, X_test)

    # Write submission file now.
    write_submission_file(ids_test, Y_test_pred, thresholded_Y_test_pred)