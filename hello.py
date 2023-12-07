import numpy as np

def main(arr:np.ndarray):
    print("Hello world!")
    print("Arr shape", arr.shape)
    print(type(arr))
    arr = np.asarray(arr)
    print(arr)
    return arr.sum()

result = main(arr)