import numpy as np

def load_firewall(return_X_y=None):
    path = "data/firewall/log2.csv"
    class_match = {"allow":0,"deny":1,"drop":2,"reset-both":3}
    class_col = 4
    with open(path) as o:
        lines = [l.strip().split(",") for l in o.readlines()[1:]]
    X = [[x for i,x in enumerate(l) if i != class_col] 
         for l in lines]
    X = np.array(X).astype(np.float32)
    y = np.array([class_match[l[class_col]] for l in lines]).astype(np.float32)
    return X,y

def load_sensorless(return_X_y=None):
    path = "data/sensorless/Sensorless_drive_diagnosis.txt"
    class_match = {str(i+1):i for i in range(11)}
    class_col = 48
    with open(path) as o:
        lines = [l.strip().split(" ") for l in o.readlines()[1:]]
    X = [[x for i,x in enumerate(l) if i != class_col] 
         for l in lines]
    X = np.array(X)
    y = np.array([class_match[l[class_col]] for l in lines]).astype(np.float32)
    return X,y

def load_scania(return_X_y=None,split="train"):
    if split == "train":
        path = "data/scania/aps_failure_training_set.csv"
    elif split == "test":
        path = "data/scania/aps_failure_test_set.csv"
    class_match = {"neg":0,"pos":1}
    class_col = 0
    with open(path) as o:
        lines = [l.strip().split(",") for l in o.readlines()[1:]]
    X = [[x for i,x in enumerate(l) if i != class_col] 
         for l in lines]
    X = np.array(X)
    # remove cols with more than 20% NA
    X = X[:,np.sum(X=="nan",axis=0)/X.shape[0] < 0.2]
    # remove entries where NA are present
    nan_idx = np.sum(X=="nan",axis=1) == 0
    X = X[nan_idx,:]
    y = np.array([class_match[l[class_col]] for l in lines]).astype(np.float32)
    y = y[nan_idx]
    return X,y

def load_sepsis(return_X_y=None,split="train"):
    if split == "train":
        path = "data/sepsis/s41598-020-73558-3_sepsis_survival_primary_cohort.csv"
    elif split == "test":
        path = "data/sepsis/s41598-020-73558-3_sepsis_survival_validation_cohort.csv"
    class_match = {"0":0,"1":1}
    class_col = 3
    with open(path) as o:
        lines = [l.strip().split(",") for l in o.readlines()[1:]]
    X = [[x for i,x in enumerate(l) if i != class_col] 
         for l in lines]
    X = np.array(X)
    y = np.array([class_match[l[class_col]] for l in lines]).astype(np.float32)
    return X,y