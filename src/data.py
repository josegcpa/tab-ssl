import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.utils.multiclass import unique_labels

open_ml_datasets = {
    "bank-marketing":{"cat_cols":[],"class_col":0},
    "california":{"cat_cols":[],"class_col":0},
    "compass":{"cat_cols":[0, 2, 3, 10,11, 12, 13, 14, 15],
               "class_col":0},
    "covertype":{"cat_cols":[
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
        50, 51, 52, 53],
                 "class_col":0},
    "credit":{"cat_cols":[],"class_col":0},
    "electricity":{"cat_cols":[2],
                   "class_col":0},
    "eye_movements":{"cat_cols":[3,4,17,22,23],
                     "class_col":0},
    "Higgs":{"cat_cols":[],"class_col":0},
    "house_16H":{"cat_cols":[],"class_col":0},
    "jannis":{"cat_cols":[],"class_col":0},
    "KDDCup09_upselling":{"cat_cols":[
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
                          "class_col":0},
    "kdd_ipums_la_97-small":{"cat_cols":[],"class_col":0},
    "MagicTelescope":{"cat_cols":[],"class_col":0},
    "MiniBooNE":{"cat_cols":[],"class_col":0},
    "phoneme":{"cat_cols":[],"class_col":0},
    "rl":{"cat_cols":[3, 4, 5, 6, 7, 8, 11],"class_col":0},
    "road-safety":{"cat_cols":[6, 22, 25],"class_col":0},
}

def load_iris():
    tmp = datasets.load_iris()
    X,y = tmp["data"],tmp["target"]
    cat_cols = []
    return X,y,cat_cols

def load_digits():
    tmp = datasets.load_digits()
    X,y = tmp["data"],tmp["target"]
    cat_cols = []
    return X,y,cat_cols

def load_wine():
    tmp = datasets.load_wine()
    X,y = tmp["data"],tmp["target"]
    cat_cols = []
    return X,y,cat_cols

def load_breast_cancer():
    tmp = datasets.load_breast_cancer()
    X,y = tmp["data"],tmp["target"]
    cat_cols = []
    return X,y,cat_cols

def load_data(path,class_match,class_col,cols_to_exclude,first_line_idx=0):
    with open(path) as o:
        lines = [l.strip().split(",") for l in o.readlines()[first_line_idx:]]
    X = [[x for i,x in enumerate(l) 
          if (i != class_col) and (i not in cols_to_exclude)]
         for l in lines]
    X = np.array(X)
    y = np.array([l[class_col] for l in lines])
    if class_match is None:
        class_match = {k:i for i,k in enumerate(unique_labels(y))}
    y = np.array([class_match[c] for c in y])
    # remove cols with more than 20% NA
    X = X[:,np.sum(X=="nan",axis=0)/X.shape[0] < 0.2]
    # remove entries where NA are present
    nan_idx = np.sum(X=="nan",axis=1) == 0
    X = X[nan_idx,:]
    y = y[nan_idx]
    X = np.float32(X)
    y = np.float32(y)
    return X,y

def load_firewall():
    path = "data/firewall/log2.csv"
    class_match = {"allow":0,"deny":1,"drop":2,"reset-both":3}
    class_col = 4
    cols_to_exclude = [0,1,2,3]
    cat_cols = []
    X,y = load_data(path,class_match,class_col,cols_to_exclude,1)
    return X,y,cat_cols

def load_sensorless():
    path = "data/sensorless/Sensorless_drive_diagnosis.csv"
    class_match = {str(i+1):i for i in range(11)}
    class_col = 48
    cols_to_exclude = []
    cat_cols = []
    X,y = load_data(path,class_match,class_col,cols_to_exclude,0)
    return X,y,cat_cols

def load_scania(split="train"):
    if split == "train":
        path = "data/scania/aps_failure_training_set.csv"
    elif split == "test":
        path = "data/scania/aps_failure_test_set.csv"
    class_match = {"neg":0,"pos":1}
    class_col = 0
    cols_to_exclude = []
    cat_cols = []
    X,y = load_data(path,class_match,class_col,cols_to_exclude,1)
    return X,y,cat_cols

def load_sepsis(split="train"):
    if split == "train":
        path = "data/sepsis/s41598-020-73558-3_sepsis_survival_primary_cohort.csv"
    elif split == "test":
        path = "data/sepsis/s41598-020-73558-3_sepsis_survival_validation_cohort.csv"
    class_match = {"0":0,"1":1}
    class_col = 3
    cols_to_exclude = []
    cat_cols = [1]
    X,y = load_data(path,class_match,class_col,cols_to_exclude,1)
    return X,y,cat_cols
    
def load_data_open_ml(name):
    path = os.path.join('data/open_ml/{}.csv'.format(name))
    class_match = None
    class_col = open_ml_datasets[name]["class_col"]
    cols_to_exclude = []
    cat_cols = open_ml_datasets[name]["cat_cols"]
    X,y = load_data(path,class_match,class_col,cols_to_exclude,1)
    return X,y,cat_cols

def load_split_unlabeled(name, p_u):
    """Function that returns both labeled and unlabeled data partitions from a fully labeled dataset.

    Args:
        name (str): name of the dataset.
        p_u (float): percentage of the dataset that will be deemed unlabeled.

    Returns:
        [numpy.array]: numpy arrays for the labeled and unlabeled data.
    """
    print("> Opening: ", name + "\n", "> % unlabeled: ", str(p_u))

    pd = path = os.path.join('data/open_ml', name)
    df = pd.read_csv(path)

    X_labeled, X_unlabeled, y, _ = train_test_split(df, random_state=42)
    return (X_labeled, y), X_unlabeled

def is_underrepresented(x:np.array,p=0.05)->np.array:
    """Function that converts a categorical array into a boolean array which
    is True at elements present in a proportion greater than p and False
    otherwise.

    Args:
        x (np.array): one-dimensional categorical array.
        p (float, optional): prevalence threshold. Defaults to 0.05.

    Returns:
        np.array: indexation array.
    """
    u,c = np.unique(x,return_counts=True)
    normalized_c = c / c.sum()
    underrepresented_u = u[normalized_c <= p]
    index_vector = np.ones_like(x,dtype=bool)
    for u in underrepresented_u:
        index_vector[x == u] = 0
    return index_vector

def exclude_underrepresented(X:np.array,cat_cols,p=0.05):
    index_vector = np.ones(X.shape[0],dtype=bool)
    for c in cat_cols:
        index_vector = index_vector * is_underrepresented(X[:,c])
    return X[index_vector,:]

supported_datasets = {
    "iris":load_iris,
    "digits":load_digits,
    "wine":load_wine,
    "breast_cancer":load_breast_cancer,
    "firewall":load_firewall,
    "sensorless":load_sensorless,
    "scania":load_scania,
    "sepsis":load_sepsis,
    # open ml datasets
    'bank-marketing': lambda: load_data_open_ml('bank-marketing'), 
    'california': lambda: load_data_open_ml('california'), 
    'compass': lambda: load_data_open_ml('compass'),
    'covertype': lambda: load_data_open_ml('covertype'),
    'credit': lambda: load_data_open_ml('credit'),
    'electricity': lambda: load_data_open_ml('electricity'),
    'eye_movements': lambda: load_data_open_ml('eye_movements'),
    'Higgs': lambda: load_data_open_ml('Higgs'),
    'house_16H': lambda: load_data_open_ml('house_16H'),
    'jannis': lambda: load_data_open_ml('jannis'),
    'KDDCup09_upselling': lambda: load_data_open_ml('KDDCup09_upselling'),
    'kdd_ipums_la_97-small': lambda: load_data_open_ml('kdd_ipums_la_97-small'),
    'MagicTelescope': lambda: load_data_open_ml('MagicTelescope'),
    'MiniBooNE': lambda: load_data_open_ml('MiniBooNE'),
    'phoneme': lambda: load_data_open_ml('phoneme'),
    'rl': lambda: load_data_open_ml('rl'),
    'road-safety': lambda: load_data_open_ml('road-safety'),
}