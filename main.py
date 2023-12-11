import numpy as np
from feature_classifier import FeatureClassifier
import pickle
from pathlib import Path
import pandas as pd


def main(testdata) -> int:
    X = np.asarray(testdata).copy()
    X = X[:, :2]
    X = (X - X.mean(0)) / X.std(0)
    classifier = pickle.load(Path("model.pickle").open("rb"))
    df = pd.DataFrame(X, columns=[0, 1])
    pred = classifier.predict(df)
    return pred


C = main(testdata)
