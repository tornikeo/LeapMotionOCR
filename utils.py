from numpy import ndarray
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
from pathlib import Path
import zipfile
import shutil
import os
from contextlib import contextmanager
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.draw import line_aa

def rasterize(points:np.ndarray, gridsize: int) -> np.ndarray:
        points = (points - points.min(0)) / (points.max(0) - points.min(0))
        points = np.ceil(points * (gridsize-1))
        points = points.astype(int)
        
        im = np.zeros((gridsize, gridsize), np.uint8)
        
        for i in range(len(points) - 1):
            x, y, _ = points[i]
            x_n, y_n, _ = points[i + 1]
            rr, cc, val = line_aa(-y, x, -y_n, x_n)
            im[rr,cc] = val * 255
        return im

def download_data():
    # If you don't have data, download it automatically
    url = "https://github.com/tornikeo/LeapMotionOCR/releases/download/0.1/digits_3d_training_data.zip"
    open('digits_3d_training_data.zip', 'wb').write(requests.get(url).content)
    with zipfile.ZipFile("digits_3d_training_data.zip","r") as zip_ref:
        zip_ref.extractall(".")
    os.remove('digits_3d_training_data.zip')
    
def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float=None,
    shuffle:bool=True,
    random_state: int=None,
) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    if random_state is not None:
        random.seed(random_state)
    
    assert len(X) == len(y), "arrays must be of same size"
    assert  0 < test_size < 1, "Test size can be in [0,1] range only"

    test_size = round(test_size * len(X))
    indices = list(range(len(X)))
    test = random.sample(indices, test_size)
    train = list(set(indices) - set(test))
    
    if shuffle:
        random.shuffle(train)
        random.shuffle(test)

    results = []
    for arr in X,y:
        for split in train,test:
            if isinstance(arr, np.ndarray):
                results.append(arr[split])
            else:
                results.append(list([arr[idx] for idx in split]))
    return results


def load_training_data(path = "training_data") -> tuple[list, list]:
    data = sorted(Path(path).glob('*.csv'))
    labels = pd.Series(data).astype(str).str.extract(r".+stroke_(\d)_.+.csv").astype(int)[0]
    data = [pd.read_csv(d, header=None) for d in data]
    return data, labels

def load_mnist() -> tuple[np.ndarray, np.ndarray]:
    mnist_x = np.load('tornikeo/mnist_digits.npy')
    mnist_y = np.load('tornikeo/mnist_targets.npy')
    return mnist_x, mnist_y        
        
def show(arr: np.ndarray) -> None:
    arr = arr.squeeze()
    if arr.ndim == 2:
        plt.matshow(arr)
    elif arr.ndim == 3:
        plt.imshow(arr)


def accuracy_score(y_true, y_pred):
    return np.isclose(y_true, y_pred).mean()

class AccuracyScorerMixin:
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

class KMeansClassifier(BaseEstimator, TransformerMixin, AccuracyScorerMixin):
    def __init__(self, n_cluster: int, max_iter=100):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ):
        assert X.ndim == 2, "Must have 2D input as X"
        assert y.squeeze().ndim == 1, "Labels must be single"
        
        n_samples, n_features = X.shape

        centers = np.random.uniform(low=X.min(),
                                    high=X.max(),
                                    size=(self.n_cluster, n_features))
        it = 0
        while True:
            if it > self.max_iter:
                print("Failed to converge")
                break
            dists = []
            for center in centers:
                dists.append(np.linalg.norm(X - center, axis=1))

            dists = np.stack(dists, 0)
            assigns = np.argmin(dists, axis=0)

            updated = centers.copy()

            for center_idx in range(self.n_cluster):
                # Check if any data points are assigned to the cluster
                if np.sum(assigns == center_idx) > 0:
                    updated[center_idx] = X[assigns == center_idx].mean(0)

            if np.allclose(updated, centers):
                break

            centers[:] = updated

            it += 1
        return self
    
class StandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass
    def fit(self, X:np.ndarray, y:np.ndarray=None):
        self.moments = [X.mean(0), X.std(0)]
        return self
    
    def transform(self, X, y=None):
        return (X - self.moments[0]) / (self.moments[1] + 1e-7)
        
class SVC(BaseEstimator, AccuracyScorerMixin):
    def __init__(self, C=1.0, gamma='auto'):
        self.C = C
        self.gamma = gamma
        self.support_vectors = None
        self.support_vector_labels = None
        self.dual_coef = None

    def rbf_kernel(self, X1, X2):
        gamma = 1.0 / (2 * X1.shape[1]) if self.gamma == 'auto' else self.gamma
        pairwise_sq_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma * pairwise_sq_dists)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Compute the Gram matrix (kernel matrix)
        K = self.rbf_kernel(X, X)

        # Quadratic programming to find the support vectors and dual coefficients
        P = np.outer(y, y) * K
        q = -np.ones(n_samples)
        G = np.vstack([-np.eye(n_samples), np.eye(n_samples)])
        h = np.hstack([np.zeros(n_samples), np.ones(n_samples) * self.C])

        solution = np.linalg.solve(P, q, overwrite_a=True, overwrite_b=True, check_finite=False)
        alpha = np.maximum(0, solution)

        # Support vectors have non-zero dual coefficients
        sv_indices = np.where(alpha > 1e-6)[0]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        self.dual_coef = alpha[sv_indices] * self.support_vector_labels
        return self

    def predict(self, X):
        # Compute the RBF kernel between test data and support vectors
        K_test = self.rbf_kernel(X, self.support_vectors)

        # Predict using the dual coefficients and support vector labels
        y_pred = np.sign(np.dot(K_test, self.dual_coef))

        return y_pred.astype(int)

class KNN(BaseEstimator, AccuracyScorerMixin):
    def __init__(self, n_neighbors:int) -> None:
        self.n_neighbors = n_neighbors
        self.xy = None
        
    def fit(self, x:np.ndarray, y:np.ndarray):
        self.xy = x.copy(), y.copy()

    def predict(self, x:np.ndarray):
        # Ensure that the model has been trained
        if self.xy is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        # Unpack training data
        x_train, y_train = self.xy
        
        # Calculate distances between each test point and all training points
        distances = np.linalg.norm(x_train[:, np.newaxis, :] - x[np.newaxis, :, :], axis=2)

        # Find the indices of the k-nearest neighbors for each test point
        nearest_neighbors_indices = np.argpartition(distances, self.n_neighbors, axis=0)[:self.n_neighbors, :]

        # Get the corresponding labels of the k-nearest neighbors
        nearest_neighbors_labels = y_train[nearest_neighbors_indices]

        # Predict the majority class for each test point
        predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 
                                          axis=0, 
                                          arr=nearest_neighbors_labels)
        return predictions


def cross_val_score(
    clf:BaseEstimator, 
    x_train:np.ndarray, 
    y_train:np.ndarray, 
    cv:int
) -> np.ndarray:
    "Returns scores over all CV splits"
    fold_size = len(x_train) // cv
    scores = np.zeros(cv, dtype='float32')
    for i in range(cv):
        start = i * fold_size
        end = start + fold_size
        x_val = x_train[start:end]
        y_val = y_train[start:end]
        
        x_train_fold = np.concatenate([x_train[:start], x_train[end:]])
        y_train_fold = np.concatenate([y_train[:start], y_train[end:]])
        
        clf.fit(x_train_fold, y_train_fold)
        
        scores[i] = clf.score(x_val, y_val)
    return scores

@contextmanager
def mute_stdout():
    stdout = sys.stdout
    sys.stdout = None
    yield
    sys.stdout = stdout
    
class PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components:int) -> None:
        super().__init__()
        self.n_components = n_components
        
    def fit(self, x:np.ndarray, y:np.ndarray=None):
        assert np.isclose(x.mean(0), 0).all() and np.isclose(x.std(0), 1).all(), \
            "Ayoo, you forgot to standartize data before going to PCA. Getcha unscaled ass outta here."
        x = x.copy()
        cov = np.cov(x, rowvar = False)
        eig_val, eig_vec = np.linalg.eig(cov)
        eig_importance = np.argsort(eig_val)[::-1] 
        self.reduction_matrix = eig_vec[:,eig_importance]
        return self
    
    def transform(self, x, y=None):
        x_reduced = np.matmul(x, self.reduction_matrix[:,:self.n_components])
        return x_reduced
    
def evaluate(
    clf,
    x_train_scale, y_train,
    x_test_scale, y_test,
) -> tuple[float, float]:
    """
    Estimates training performance with CV.
    Also returns test performance. 
    """
    cv_score = cross_val_score(clf, 
                    x_train_scale, y_train, 
                    cv = 4)
    clf.fit(x_train_scale, y_train)
    test_acc = accuracy_score(y_test, clf.predict(x_test_scale))
    return cv_score, test_acc