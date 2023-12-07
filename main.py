import numpy as np
from feature_classifier import FeatureClassifier
import pickle
from pathlib import Path
import pandas as pd

def main(testdata: np.ndarray):
    X = np.asarray(testdata).copy()
    X = X[:, :2]
    X = (X - X.mean()) / X.std()
    classifier = pickle.load(Path('model.pickle').open('rb'))
    df = pd.DataFrame(X, columns=[0, 1])
    return classifier.predict(df)

C = main(testdata)
