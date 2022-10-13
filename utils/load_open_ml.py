import numpy as np
import pandas as pd
import sys
import os


def load_data(name):
    print("> Opening: ", name)

    path = os.path.join('../data/open_ml', name)
    df = pd.read_csv(path)

    y = df['class']
    X = df.drop(columns=['class'])

    return X, y


