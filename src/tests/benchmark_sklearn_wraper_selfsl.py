import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate,StratifiedKFold
from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import (
    load_iris,load_diabetes,load_digits,load_linnerud,load_wine,
    load_breast_cancer)

from typing import List

from ..data_generator import AutoDataset
from ..sklearn_wrappers import SKLearnSelfSLVIME
from ..data import (load_sepsis,load_scania,load_firewall,load_sensorless)

dataset_loaders = [
    load_iris,
    load_digits,
    load_wine,
    load_breast_cancer,
    load_firewall,
    load_sensorless,
    load_sepsis,
    load_scania
    ]

dataset_loaders_split = [
    load_sepsis,load_scania
]

random_state = 42
n_folds = 5
ss = 1000
ss = None

for dataset_loader in dataset_loaders:
    X,y = dataset_loader(return_X_y=True)
    if ss is not None:
        rs = np.random.choice(X.shape[0],np.minimum(ss,X.shape[0]))
        X,y = X[rs],y[rs]
    n_samples,n_features = X.shape

    if dataset_loader.__name__ in ["fetch_california_housing","load_sensorless"]:
        ct = 10
    elif dataset_loader.__name__ in ["load_digits","load_scania"]:
        ct = 0
    elif dataset_loader.__name__ in ["load_firewall"]:
        ct = 4
    else:
        ct = 0.1

    if n_features // 2 < 5:
        O = n_features
    else:
        O = n_features
    vime = SKLearnSelfSLVIME(
        [*[n_features for _ in range(1)],O],
        [n_features for _ in range(1)],
        [n_features for _ in range(1)],
        learning_rate=0.01,
        mask_p=0.2,
        alpha=2.0,
        max_iter=2000,
        n_iter_no_change=50,
        batch_size=n_samples//5,
        optimizer="rmsprop",
        optimizer_params=[("weight_decay",0.005)],
        act_fn="relu",
        batch_norm=True,
        verbose=True,
        cat_thresh=ct)
    
    standard_rf = Pipeline([
        ("preprocessing",AutoDataset(cat_thresh=ct)),
        ("estimator",RandomForestClassifier(random_state=random_state))])
    vime_rf = Pipeline([
        ("preprocessing",vime),
        ("estimator",RandomForestClassifier(random_state=random_state))])

    print(dataset_loader.__name__,n_samples,n_features)
    if len(np.unique(y)) > 2:
        metric = "f1_macro"
    else:
        metric = "f1"
    
    rf_scores = cross_validate(
        standard_rf,
        X,y,
        cv=StratifiedKFold(n_folds,shuffle=True,random_state=random_state),
        scoring=metric,
        n_jobs=1)
    vime_rf_scores = cross_validate(
        vime_rf,
        X,y,
        cv=StratifiedKFold(n_folds,shuffle=True,random_state=random_state),
        scoring=metric,
        n_jobs=1)

    print("\tClassifier scores = {} (std.err.={})".format(
        np.mean(rf_scores["test_score"]),
        np.std(rf_scores["test_score"])))
    print("\tVIME + classifier scores = {} (std.err.={})".format(
          np.mean(vime_rf_scores["test_score"]),
          np.std(vime_rf_scores["test_score"])))
