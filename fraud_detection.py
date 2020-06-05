import os
import time
import warnings

warnings.filterwarnings('ignore')

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import regularizers
from keras.layers import Dense, Input, Dropout, Lambda
from keras.models import Model, Sequential
from keras import backend as K
from keras.losses import mse
import optuna
from sklearn import ensemble, model_selection, preprocessing, decomposition, cluster, metrics, linear_model, pipeline

from imblearn import under_sampling, combine

import random as python_random
import xgboost
import math

seed = 2442

np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED']=str(seed)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

def load_dataset(path, target_column="target", train_size=0.8, split=False, stratify=False, seed=666):
    if(split):
        df = pd.read_csv(path, engine="c")

        x = df.drop(target_column, axis=1)
        y = df[target_column].values

        if(stratify):
            x_train, x_dev_test, y_train, y_dev_test = model_selection.train_test_split(x, y, train_size=0.8, stratify=y, random_state=seed)
            x_dev, x_test, y_dev, y_test = model_selection.train_test_split(x_dev_test, y_dev_test, train_size=0.5, stratify=y_dev_test, random_state=seed)
        else:
            x_train, x_dev_test, y_train, y_dev_test = model_selection.train_test_split(x, y, train_size=0.8, random_state=seed)
            x_dev, x_test, y_dev, y_test = model_selection.train_test_split(x_dev_test, y_dev_test, train_size=0.5, random_state=seed)

        return x_train, y_train, x_dev, y_dev, x_test, y_test
    else:
        df = pd.read_csv(path, engine="c")

        x = df.drop(target_column, axis=1)
        y = df[target_column].values

        return x,y

def normalize_data(x_train, x_dev=pd.DataFrame(), x_test=pd.DataFrame()):
    model_minmax = preprocessing.MinMaxScaler()

    x_train = model_minmax.fit_transform(x_train.values)
    if(not x_dev.empty):
        x_dev = model_minmax.transform(x_dev.values)
    if(not x_test.empty):
        x_test = model_minmax.transform(x_test.values)

    return x_train, x_dev, x_test


def autoenconder(x_train, y_train, x_dev=pd.DataFrame(), x_test=pd.DataFrame(), nb_samples=100000, target_class=0):
    x_train_class = x_train[y_train == target_class]

    x_train_sample = x_train_class[:nb_samples]
    x_train_sample = x_train_class

    model = Sequential()
    model.add(Dense(25, activation='tanh', activity_regularizer=regularizers.l1(10e-5), input_shape=(x_train_sample.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='relu', name="bottleneck"))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(25, activation='tanh'))
    model.add(Dense(x_train_sample.shape[1], activation='relu'))

    cb_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, mode="min", restore_best_weights=True)

    model.compile(optimizer="adam", loss="mse")

    model.fit(x_train_sample, x_train_sample, batch_size=1024, epochs=70, shuffle=True, validation_split=0.2, verbose=0, callbacks=[cb_early_stopping])

    encoder = Model(model.input, model.get_layer('bottleneck').output)

    x_train = encoder.predict(x_train)
    if(x_dev.shape[0]):
        x_dev = encoder.predict(x_dev)
    if(x_test.shape[0]): 
        x_test = encoder.predict(x_test)

    return x_train, x_dev, x_test

def under_sample_train(x_train, y_train, random=False, seed=666):
    if(random):
        model_under_sample = under_sampling.RandomUnderSampler(random_state=seed)
        x_train, y_train = model_under_sample.fit_resample(x_train, y_train)
    else:
        model_under_sample = under_sampling.NearMiss(version=2, random_state=seed, n_jobs=-1)
        x_train, y_train = model_under_sample.fit_resample(x_train, y_train)
    return x_train, y_train

def calculate_inertia(x_train_pca, seed=666):
    inertia_list = []
    for n in range(2, 21):
        model_kmeans = cluster.KMeans(n_clusters=n, init="k-means++", random_state=seed, n_jobs=-1)
        model_kmeans.fit(x_train_pca)
        inertia_list.append(model_kmeans.inertia_)
    return inertia_list

def optimal_number_clusters(inertia_list):
    x1, y1 = 2, inertia_list[0]
    x2, y2 = 20, inertia_list[-1]

    distances = []
    for i in range(len(inertia_list)):
        x0 = i+2
        y0 = inertia_list[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 2

def plot_kmeans(x_train, n, name_image, seed=666):
    plt.clf()
    model_pca = decomposition.PCA(n_components=2, random_state=seed)
    x_train_pca = model_pca.fit_transform(x_train)

    model_kmeans = cluster.KMeans(n_clusters=n, init="k-means++", random_state=seed, n_jobs=-1)
    model_kmeans.fit(x_train_pca)

    y_kmeans = model_kmeans.predict(x_train_pca)
    cluster_centers = model_kmeans.cluster_centers_

    plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=y_kmeans, s=50, cmap='viridis')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=200, alpha=0.5)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    path_img = 'doc/img/' + name_image + '.png'
    if(os.path.exists(path_img) == False):
        plt.savefig(path_img)


def subtract_samples(x, y, cluster_centers):
    sample_list = []
    target_list = []
    for i in range(len(x)):
        for centroid_sample in cluster_centers:
            sample_list.append(np.subtract(x[i], np.array(centroid_sample)))
            target_list.append(y[i])
    return np.array(sample_list), np.array(target_list)

def clustering(x_train, y_train, target_class=0, seed=666, plot=False, name_image="kmeans_plot"):
    x_train_class = x_train[y_train == target_class]

    inertia_list = calculate_inertia(x_train_class, seed)
    n = optimal_number_clusters(inertia_list)

    model_kmeans = cluster.KMeans(n_clusters=n, init="k-means++", random_state=seed, n_jobs=-1)
    model_kmeans.fit(x_train_class)

    y_kmeans = model_kmeans.predict(x_train_class)
    cluster_centers = model_kmeans.cluster_centers_

    if(plot):
        plot_kmeans(x_train_class, n, name_image, seed)

    return cluster_centers, n

def sum_predictions(pred_proba, n):
    real_pred = []
    for i in range(0, pred_proba.shape[0], n):
        pred_matrix = []
        for j in range(n):
            pred_matrix.append(pred_proba[i+j])
        pred_tmp = np.argmax(np.sum(np.array(pred_matrix), axis=0)) 
        real_pred.append(pred_tmp)

    return real_pred

def mean_predictions(pred_proba, n):
    real_pred = []
    for i in range(0, pred_proba.shape[0], n):
        pred_matrix = []
        for j in range(n):
            pred_matrix.append(pred_proba[i+j])
        pred_tmp = np.argmax(np.mean(np.array(pred_matrix), axis=0)) 
        real_pred.append(pred_tmp)

    return real_pred


def vote_predictions(pred_proba, n, threshold):
    real_pred = []
    for i in range(0, pred_proba.shape[0], n):
        pred_matrix = []
        for j in range(n):
            pred_matrix.append(pred_proba[i+j])
        pred_thresh = preprocessing.binarize(np.array(pred_matrix)[:, 1].reshape(1, -1), threshold)[0]
        pred_tmp = max(set(list(pred_thresh)), key=list(pred_thresh).count)
        real_pred.append(pred_tmp)

    return real_pred

def objective_xgboost(trial, x_train, y_train, seed=666):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 7, 30),
        "subsample": trial.suggest_discrete_uniform('subsample', 0.1, 1.0, 0.1),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.1),
        "reg_alpha": trial.suggest_int("reg_alpha", 0, 300),
        "reg_lambda": trial.suggest_int("reg_lambda", 0, 300),
        'min_child_weight' : trial.suggest_int("min_child_weight", 0, 300),
    }
    
    metric_list = []
    
    kf = model_selection.StratifiedKFold(n_splits=3, shuffle=False, random_state=seed)
    
    for train_index, test_index in kf.split(x_train, y_train):
        x_train_gs, x_test_gs = x_train[train_index], x_train[test_index]
        y_train_gs, y_test_gs = y_train[train_index], y_train[test_index]

        model_xgb = xgboost.XGBClassifier(**param, objective="binary:logistic", random_state=seed, n_jobs=-1, verbosity=0)
        model_xgb.fit(x_train_gs, y_train_gs)

        pred = model_xgb.predict(x_test_gs)
        pred_proba = model_xgb.predict_proba(x_test_gs)

        metric_list.append(metrics.f1_score(y_test_gs, pred))
    
    return np.mean(metric_list)

def tuning_hyperparameters(x_train, y_train, seed=666):
    study = optuna.create_study(direction='maximize')
    optuna.logging.disable_default_handler()
    study.optimize(lambda trial: objective_xgboost(trial, x_train, y_train, seed), n_trials=100, n_jobs=-1)
    trial = study.best_trial
    params = trial.params

    return params

def false_positive_rate_score(confusion_matrix):
    return confusion_matrix[0][1]/(confusion_matrix[0][0]+confusion_matrix[0][1])

def false_negative_rate_score(confusion_matrix):
    return confusion_matrix[1][0]/(confusion_matrix[1][1]+confusion_matrix[1][0])

def optimize_threshold(x_train, y_train, model):
    pred_proba_threshold_list = []
    y_test_threshold_list = []

    skf_splitter_threshold = model_selection.StratifiedKFold(n_splits=3)
    
    for train_index_threshold, test_index_threshold in skf_splitter_threshold.split(x_train, y_train):
        x_train_threshold, x_test_threshold = x_train[train_index_threshold], x_train[test_index_threshold]
        y_train_threshold, y_test_threshold = y_train[train_index_threshold], y_train[test_index_threshold]

        model.fit(x_train_threshold, y_train_threshold)
        pred_proba_threshold = model.predict_proba(x_test_threshold)
        
        for i in range(len(y_test_threshold)):
            pred_proba_threshold_list.append(pred_proba_threshold[i,1])
            y_test_threshold_list.append(y_test_threshold[i])
    
    thresholds = np.arange(0, 1, 0.001)
    proba_threshold = np.array(pred_proba_threshold_list).reshape(1, -1)
    scores = [metrics.precision_score(y_test_threshold_list, preprocessing.binarize(proba_threshold, t)[0]) for t in thresholds]
     
    best_index = np.argmax(scores)
    threshold_final = thresholds[best_index]

    return threshold_final

def execute_pipeline1(x_train, y_train, x_test, y_test, name_image, seed=666):
    x_train, x_test, _ = normalize_data(x_train, x_test)
    x_train, x_test, _ = autoenconder(x_train, y_train, x_test)
    x_train, y_train = under_sample_train(x_train, y_train, random=True, seed=seed)
    cluster_centers_list, n = clustering(x_train, y_train, seed=seed, plot=True, name_image=name_image)

    x_train, y_train = subtract_samples(x_train, y_train, cluster_centers_list)
    x_test, y_test = subtract_samples(x_test, y_test, cluster_centers_list)

    params = tuning_hyperparameters(x_train, y_train, seed)

    model = xgboost.XGBClassifier(**params, objective="binary:logistic", random_state=seed, n_jobs=-1, verbosity=0)
    model.fit(x_train, y_train)

    pred_test = model.predict(x_test)
    pred_proba_test = model.predict_proba(x_test)

    threshold_final = optimize_threshold(x_train, y_train, model)
    pred_test_real = vote_predictions(pred_proba_test, n, threshold_final)

    return pred_test_real

def execute_pipeline2(x_train, y_train, x_test, y_test, name_image,seed=666):
    x_train, x_test, _ = normalize_data(x_train, x_test)
    x_train, x_test, _ = autoenconder(x_train, y_train, x_test)
    x_train, y_train = under_sample_train(x_train, y_train, random=True, seed=seed)
    cluster_centers_list, n = clustering(x_train, y_train, seed=seed, plot=True, name_image=name_image)

    x_train, y_train = subtract_samples(x_train, y_train, cluster_centers_list)
    x_test, y_test = subtract_samples(x_test, y_test, cluster_centers_list)

    params = tuning_hyperparameters(x_train, y_train, seed)

    model = xgboost.XGBClassifier(**params, objective="binary:logistic", random_state=seed, n_jobs=-1, verbosity=0)
    model.fit(x_train, y_train)

    pred_test = model.predict(x_test)
    pred_proba_test = model.predict_proba(x_test)

    pred_test_real = sum_predictions(pred_proba_test, n)

    return pred_test_real

def execute_pipeline3(x_train, y_train, x_test, y_test, name_image, seed=666):
    x_train, x_test, _ = normalize_data(x_train, x_test)
    x_train, x_test, _ = autoenconder(x_train, y_train, x_test)
    x_train, y_train = under_sample_train(x_train, y_train, random=True, seed=seed)
    cluster_centers_list, n = clustering(x_train, y_train, seed=seed, plot=True, name_image=name_image)

    x_train, y_train = subtract_samples(x_train, y_train, cluster_centers_list)
    x_test, y_test = subtract_samples(x_test, y_test, cluster_centers_list)

    params = tuning_hyperparameters(x_train, y_train, seed)

    model = xgboost.XGBClassifier(**params, objective="binary:logistic", random_state=seed, n_jobs=-1, verbosity=0)
    model.fit(x_train, y_train)

    pred_test = model.predict(x_test)
    pred_proba_test = model.predict_proba(x_test)

    pred_test_real = mean_predictions(pred_proba_test, n)

    return pred_test_real

def execute_pipeline4(x_train, y_train, x_test, y_test, seed=666):
    x_train, x_test, _ = normalize_data(x_train, x_test)
    x_train, x_test, _ = autoenconder(x_train, y_train, x_test)
    x_train, y_train = under_sample_train(x_train, y_train, random=True, seed=seed)

    params = tuning_hyperparameters(x_train, y_train, seed)

    model = xgboost.XGBClassifier(**params, objective="binary:logistic", random_state=seed, n_jobs=-1, verbosity=0)
    model.fit(x_train, y_train)

    pred_test = model.predict(x_test)
    pred_proba_test = model.predict_proba(x_test)

    threshold_final = optimize_threshold(x_train, y_train, model)

    pred_test_real = preprocessing.binarize(pred_proba_test[:,1].reshape(-1, 1), threshold_final)

    return pred_test_real

def cross_validation(x, y, n, pipeline, name_image="kmeans_plot", seed=666):
    f1_list = []
    recall_list = []
    precision_list = []
    fpr_list = []
    fnr_list = []

    kf = model_selection.StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)

    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        predict_test = pipeline(x_train, y_train, x_test, y_test, name_image, seed)

        cfn_matrix = metrics.confusion_matrix(y_test, predict_test)
        f1 = metrics.f1_score(y_test, predict_test)
        recall = metrics.recall_score(y_test, predict_test)
        precision = metrics.precision_score(y_test, predict_test)
        fpr = false_positive_rate_score(metrics.confusion_matrix(y_test, predict_test))
        fnr = false_negative_rate_score(metrics.confusion_matrix(y_test, predict_test))

        f1_list.append(f1)
        recall_list.append(recall)
        precision_list.append(precision)
        fpr_list.append(fpr)
        fnr_list.append(fnr)

    print("F1: %f +- %f." % (np.mean(f1_list, axis=0), np.std(f1_list, axis=0)))
    print("Recall: %f +- %f." % (np.mean(recall_list, axis=0), np.std(recall_list, axis=0)))
    print("Precision: %f +- %f." % (np.mean(precision_list, axis=0), np.std(precision_list, axis=0)))
    print("False Positive Rate: %f +- %f." % (np.mean(fpr_list, axis=0), np.std(fpr_list, axis=0)))
    print("False Negative Rate: %f +- %f." % (np.mean(fnr_list, axis=0), np.std(fnr_list, axis=0)))

x,y = load_dataset("dataset/creditcard.csv", target_column="Class")

cross_validation(x, y, 5, execute_pipeline1, name_image="kmeans_pipeline1", seed=seed)
cross_validation(x, y, 5, execute_pipeline2, name_image="kmeans_pipeline2", seed=seed)
cross_validation(x, y, 5, execute_pipeline3, name_image="kmeans_pipeline3", seed=seed)
cross_validation(x, y, 5, execute_pipeline4, seed=seed)
