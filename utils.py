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
    *arrays,
    test_size=None,
    random_state=None,
):
    for arr in arrays:
        if random_state is not None:
            random.seed(random_state)
        size = len(arr)
        if test_size < 1:
            test_size = round(size * test_size)
        idx = random.sample(range(size), test_size)
        arr_sampled = arr[idx]
        return arr_sampled