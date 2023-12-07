import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import random
import utils
import argparse
from scipy.special import kl_div
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

LINE_BINS = 12
ANGLE_BINS = 12

def featurize_lines(df: pd.DataFrame) -> np.ndarray:
    """
    Converts the sequence of x,y,z points into LINE_BINS-sized
    array of features. Each element in this array represents
    how much, on average the line has traveled in a particular direction in
    space, in total.
    
    args:
        df: Pandas dataframe with 3 columns (x,y,z)
    returns:
        np.ndarray: (LINE_BINS,), positive array that encodes direction information
    """
    features = []
    buckets = np.zeros((LINE_BINS))

    for i in range(df.shape[0]-1):
        dir = df.iloc[i+1] - df.iloc[i]
        angle = np.arctan2(dir.iloc[1], dir.iloc[0])
        bucket = bucketize(angle, LINE_BINS, (-np.pi, np.pi))
        magnitude = np.linalg.norm(dir)
        buckets[bucket] += magnitude
    res = buckets / buckets.sum()
    print(res)
    return res

def featurize(df: pd.DataFrame):
    """
    
    """
    angle_features = featurize_point_vector(df)
    line_features = featurize_lines(df)
    # line_features = []
    res = np.concatenate((line_features, angle_features))
    return res

def bucketize(x, num_buckets, range):
    res = int((x - range[0]) / (range[1] - range[0]) * num_buckets)
    if res == num_buckets:
        res -= 1
    return res

def featurize_point_vector(df: pd.DataFrame):
    features = []
    buckets = np.zeros((ANGLE_BINS))
    for i in range(df.shape[0]-1):
        angle = np.arctan2(df.iloc[i, 1], df.iloc[i, 0])
        bucket = bucketize(angle, ANGLE_BINS, (-np.pi, np.pi))
        magnitude = np.linalg.norm(df.iloc[i])
        buckets[bucket] += magnitude

    # res = np.histogram(features, bins=ANGLE_BINS, range=(-np.pi, np.pi))[0]
    res = buckets / buckets.sum()
    return res

if __name__ == "__main__":
    X, y = utils.load_training_data("training_data")
    for i in range(len(X)):
        X[i].drop(2, axis=1, inplace=True)
        X[i] = (X[i] - X[i].mean()) / X[i].std()
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    num_clusters = [
        2,  # 0
        4,  # 1
        3,  # 2
        2,  # 3
        2,  # 4
        2,  # 5
        2,  # 6
        3,  # 7
        2,  # 8
        2,  # 9
    ]

    cluster_mapping = [i for i in range(10) for _ in range(num_clusters[i])]
    print(cluster_mapping)
    fvs = []

    for num in range(10):
        features = []
        for i, df in enumerate(train_X):
            if train_y.iloc[i] != num:
                continue
            features.append(featurize(df))

        km = KMeans(n_clusters=num_clusters[num], n_init=20)
        km.fit(features)
        fvs += km.cluster_centers_.tolist()
        for i in range(num_clusters[num]):
            print(f"Num: {num},Cluster {i}: {km.labels_.tolist().count(i)}")

    correct = 0
    incorrect = 0

    y_true = []
    y_pred = []

    for i, df in enumerate(test_X):
        test_features = featurize(df)

        scores = []
        for ti, template in enumerate(fvs):
            # score = chisquare(tmp, template)
            score = np.sum(kl_div(test_features, template))
            # scores.append(score.pvalue)
            scores.append(score)

        selected = cluster_mapping[np.argmin(scores)]
        # selected = np.argmax(scores)//2
        # print(f"i: {i}, Num: {num}, Estimate: {selected}")
        y_true.append(test_y.iloc[i])
        y_pred.append(selected)

    print(classification_report(y_true, y_pred))

    confusion_matrix(y_true, y_pred)