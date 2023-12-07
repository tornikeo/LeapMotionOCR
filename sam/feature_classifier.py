import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from scipy.special import kl_div

LINE_BINS = 12
ANGLE_BINS = 12
BINS = LINE_BINS + ANGLE_BINS

def bucketize(x, num_buckets, range_):
    res = int((x - range_[0]) / (range_[1] - range_[0]) * num_buckets)
    if res == num_buckets:
        res -= 1
    return res
    
def featurize_point_vector(df: pd.DataFrame):
    buckets = np.zeros((ANGLE_BINS))
    for i in range(df.shape[0]-1):
        angle = np.arctan2(df.iloc[i, 1], df.iloc[i, 0])
        bucket = bucketize(angle, ANGLE_BINS, (-np.pi, np.pi))
        magnitude = np.linalg.norm(df.iloc[i])
        buckets[bucket] += magnitude

    # res = np.histogram(features, bins=ANGLE_BINS, range=(-np.pi, np.pi))[0]
    res = buckets / buckets.sum()
    return res

def featurize_lines(df: pd.DataFrame):
    buckets = np.zeros((LINE_BINS))

    for i in range(df.shape[0]-1):
        dir_vec = df.iloc[i+1] - df.iloc[i]
        angle = np.arctan2(dir_vec.iloc[1], dir_vec.iloc[0])

        bucket = bucketize(angle, LINE_BINS, (-np.pi, np.pi))
        magnitude = np.linalg.norm(dir_vec)
        buckets[bucket] += magnitude

    # res = np.histogram(features, bins=LINE_BINS, range=(-np.pi, np.pi))[0]
    # res = res / res.sum()
    res = buckets / buckets.sum()

    return res

def featurize(df: pd.DataFrame):
    angle_features = featurize_point_vector(df)
    line_features = featurize_lines(df)
    # line_features = []
    res = np.concatenate((line_features, angle_features))
    return res


class FeatureClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.fvs = []
        self.cluster_mapping = []

    def fit(self, X, y):
        # drop third coordinate if present
        if X[0].shape[1] == 3:
            for i in range(len(X)):
                X[i].drop(2, axis=1, inplace=True)

        for i in range(len(X)):
            X[i] = (X[i] - X[i].mean()) / X[i].std()

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

        self.cluster_mapping = [i for i in range(10) for _ in range(num_clusters[i])]

        self.fvs = []

        for num in range(10):
            features = []
            for i, df in enumerate(X):
                if y.iloc[i] != num:
                    continue
                features.append(featurize(df))

            km = KMeans(n_clusters=num_clusters[num], n_init=20)
            km.fit(features)
            self.fvs += km.cluster_centers_.tolist()
            for i in range(num_clusters[num]):
                print(f"Num: {num},Cluster {i}: {km.labels_.tolist().count(i)}")

        # Return the classifier
        return self
        
    def predict(self, X):
        if X[0].shape[1] == 3:
            for i in range(len(X)):
                X[i].drop(2, axis=1, inplace=True)

        for i in range(len(X)):
            X[i] = (X[i] - X[i].mean()) / X[i].std()

        preds = []
        for df in X:
            test_features = featurize(df)

            scores = []
            for template in self.fvs:
                score = np.sum(kl_div(test_features, template))
                scores.append(score)

            selected = self.cluster_mapping[np.argmin(scores)]
            preds.append(selected)

        return preds
