import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from scipy.special import kl_div

LINE_BINS = 12
ANGLE_BINS = 12

def bucketize(x, num_buckets, range_):
    res = int((x - range_[0]) / (range_[1] - range_[0]) * num_buckets)
    if res == num_buckets:
        res -= 1
    return res
    
def featurize_point_vector(df: pd.DataFrame, bin_count):
    buckets = np.zeros(bin_count)
    for i in range(df.shape[0]-1):
        angle = np.arctan2(df.iloc[i, 1], df.iloc[i, 0])
        bucket = bucketize(angle, bin_count, (-np.pi, np.pi))
        magnitude = np.linalg.norm(df.iloc[i])
        buckets[bucket] += magnitude

    res = buckets / buckets.sum()
    return res

def featurize_lines(df: pd.DataFrame, bin_count):
    buckets = np.zeros(bin_count)

    for i in range(df.shape[0]-1):
        dir_vec = df.iloc[i+1] - df.iloc[i]
        angle = np.arctan2(dir_vec.iloc[1], dir_vec.iloc[0])

        bucket = bucketize(angle, bin_count, (-np.pi, np.pi))
        magnitude = np.linalg.norm(dir_vec)
        buckets[bucket] += magnitude

    res = buckets / buckets.sum()

    return res

def featurize(df: pd.DataFrame, line_bin_count, angle_bin_count):
    angle_features = featurize_point_vector(df, angle_bin_count)
    line_features = featurize_lines(df, line_bin_count)
    res = np.concatenate((line_features, angle_features))
    return res


class FeatureClassifier(BaseEstimator, ClassifierMixin):
    DEFAULT_CLUSTER_COUNT = (
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
    )

    def __init__(self, line_bin_count=LINE_BINS, angle_bin_count=ANGLE_BINS, cluster_count=DEFAULT_CLUSTER_COUNT):
        self.fvs = []
        self.cluster_mapping = []
        self.line_bin_count = line_bin_count
        self.angle_bin_count = angle_bin_count
        self.cluster_count = cluster_count

    def fit(self, X, y):


        self.cluster_mapping = [i for i in range(10) for _ in range(self.cluster_count[i])]

        self.fvs = []

        for num in range(10):
            features = []
            for i, df in enumerate(X):
                if y.iloc[i] != num:
                    continue
                features.append(featurize(df, self.line_bin_count, self.angle_bin_count))

            km = KMeans(n_clusters=self.cluster_count[num], n_init=20)
            km.fit(features)
            self.fvs += km.cluster_centers_.tolist()
            for i in range(self.cluster_count[num]):
                print(f"Num: {num},Cluster {i}: {km.labels_.tolist().count(i)}")

        return self
        
    def predict(self, X):
        preds = []
        for df in X:
            test_features = featurize(df, self.line_bin_count, self.angle_bin_count)

            scores = []
            for template in self.fvs:
                score = np.sum(kl_div(test_features, template))
                scores.append(score)

            selected = self.cluster_mapping[np.argmin(scores)]
            preds.append(selected)

        return preds
