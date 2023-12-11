import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.special import kl_div
from utils import KMeans


def bucketize(x: float, num_buckets: int, range_: tuple[int, int]) -> int:
    """
    Bucketizes a value based on a given range into a specified number of buckets.

    Args:
        x (float): The value to be bucketized.
        num_buckets (int): The number of buckets to divide the range into.
        range_ (tuple[int, int]): The range of values to consider for bucketization.

    Returns:
        int: The bucket index where the value falls into.
    """
    res = int((x - range_[0]) / (range_[1] - range_[0]) * num_buckets)
    if res == num_buckets:
        res -= 1
    return res


def featurize_points(df: pd.DataFrame, bin_count: int) -> np.ndarray:
    """
    Featurizes a points by calculating the magnitude of each point vector and
    distributing them into bins based on their angle.

    Args:
        df (pd.DataFrame): The input DataFrame containing the point vector.
        bin_count (int): The number of bins to distribute the points into.

    Returns:
        np.ndarray: The featurized point vector as an array of normalized bin values.
    """
    buckets = np.zeros(bin_count)
    for i in range(df.shape[0] - 1):
        angle = np.arctan2(df.iloc[i, 1], df.iloc[i, 0])
        bucket = bucketize(angle, bin_count, (-np.pi, np.pi))
        magnitude = np.linalg.norm(df.iloc[i])
        buckets[bucket] += magnitude

    res = buckets / buckets.sum()
    return res


def featurize_lines(df: pd.DataFrame, bin_count: int) -> np.ndarray:
    """
    Featurizes the lines in a DataFrame by calculating the angle and magnitude of each line segment.

    Args:
        df (pd.DataFrame): The DataFrame containing the line segments.
        bin_count (int): The number of bins to divide the angle range into.

    Returns:
        np.ndarray: The featurized representation of the lines.
    """

    buckets = np.zeros(bin_count)

    for i in range(df.shape[0] - 1):
        dir_vec = df.iloc[i + 1] - df.iloc[i]
        angle = np.arctan2(dir_vec.iloc[1], dir_vec.iloc[0])

        bucket = bucketize(angle, bin_count, (-np.pi, np.pi))
        magnitude = np.linalg.norm(dir_vec)
        buckets[bucket] += magnitude

    res = buckets / buckets.sum()
    return res


def featurize(
    df: pd.DataFrame, line_bin_count: int, angle_bin_count: int
) -> np.ndarray:
    """
    Featurizes the given DataFrame by extracting line and angle features.

    Args:
        df (pd.DataFrame): The input DataFrame containing data points.
        line_bin_count (int): The number of bins to use for line features.
        angle_bin_count (int): The number of bins to use for angle features.

    Returns:
        np.ndarray: The concatenated line and angle features as a numpy array.
    """
    angle_features = featurize_points(df, angle_bin_count)
    line_features = featurize_lines(df, line_bin_count)
    res = np.concatenate((line_features, angle_features))
    return res


class FeatureClassifier(BaseEstimator, ClassifierMixin):
    """
    Classify digits using extracted features.
    """

    LINE_BINS = 12  # length of the line segments feature vector
    ANGLE_BINS = 12  # length of the angle feature vector
    # number of clusters for each digit
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

    def __init__(
        self,
        line_bin_count=LINE_BINS,
        angle_bin_count=ANGLE_BINS,
        cluster_count=DEFAULT_CLUSTER_COUNT,
    ):
        self.fvs = []
        self.cluster_mapping = []
        self.line_bin_count = line_bin_count
        self.angle_bin_count = angle_bin_count
        self.cluster_count = cluster_count

    def fit(self, X, y):
        """
        Fit the classifier to the training data.

        Parameters:
        - X (list): List of input data.
        - y (list): List of target labels.

        Returns:
        - self (FeatureClassifier): The fitted classifier.
        """
        self.cluster_mapping = [
            i for i in range(10) for _ in range(self.cluster_count[i])
        ]
        self.fvs = []
        for num in range(10):
            features = []
            for i, df in enumerate(X):
                if y.iloc[i] != num:
                    continue
                features.append(
                    featurize(df, self.line_bin_count, self.angle_bin_count)
                )

            km = KMeans(n_clusters=self.cluster_count[num], n_init=20)
            km.fit(np.array(features))
            self.fvs += km.centroids.tolist()

        return self

    def predict(self, X):
        """
        Predict the labels for the input data.

        Parameters:
        - X (list): List of input data.

        Returns:
        - preds (list): List of predicted labels.
        """
        test_features = featurize(X, self.line_bin_count, self.angle_bin_count)

        scores = []
        for template in self.fvs:
            score = np.sum(kl_div(test_features, template))
            scores.append(score)

        selected = self.cluster_mapping[np.argmin(scores)]

        return int(selected)
