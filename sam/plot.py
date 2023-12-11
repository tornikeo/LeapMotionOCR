from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
from pathlib import Path
import zipfile

df = pd.read_csv("training_data/stroke_1_0008.csv", header=None)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot(df[0], df[1], df[2])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
