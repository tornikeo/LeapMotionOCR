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

def download_data():
    # If you don't have data, download it automatically
    url = "https://github.com/tornikeo/LeapMotionOCR/releases/download/0.1/digits_3d_training_data.zip"
    open('digits_3d_training_data.zip', 'wb').write(requests.get(url).content)
    with zipfile.ZipFile("digits_3d_training_data.zip","r") as zip_ref:
        zip_ref.extractall(".")
    os.remove('digits_3d_training_data.zip')
    
def train_test_split(
    X: np.ndarray,y: np.ndarray,
    test_size: float=None,
    random_state: int=None,
) -> tuple[np.ndarray,np.ndarray]:
    if random_state is not None:
        random.seed(random_state)
    assert len(X) == len(y), "arrays must be of same size"
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray), "X must be a numpy array"
    assert test_size > 0 and abs(test_size) < 1, "Test size can be in [0,1] range only"
    
    test_size = round(test_size * len(X))
    idx = random.sample(range(len(X)), test_size)
    return X[idx], y[idx]

    
