from ..data import load_firewall
from ..sklearn_wrappers import SKLearnSelfSLContrastive
from src.modules import SelfSLAE, SelfSLAE
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate,StratifiedKFold
import numpy as np

X, y = load_firewall()

print('shape X: ', X.shape)

model = SKLearnSelfSLContrastive()

random_state = 42

standard_rf = Pipeline([
    ("estimator", RandomForestClassifier(random_state=random_state))])
contrastive_rf = Pipeline([
    ("preprocessing", model),
    ("estimator", RandomForestClassifier(random_state=random_state))])

if len(np.unique(y)) > 2:
    metric = "f1_macro"
else:
    metric = "f1"

n_folds=3

rf_scores = cross_validate(
    standard_rf,
    X, y,
    cv=StratifiedKFold(n_folds, shuffle=True, random_state=random_state),
    scoring=metric,
    n_jobs=1)
contrastive_rf_scores = cross_validate(
    contrastive_rf,
    X, y,
    cv=StratifiedKFold(n_folds, shuffle=True, random_state=random_state),
    scoring=metric,
    n_jobs=1)

print("\tClassifier scores = {} (std.err.={})".format(
    np.mean(rf_scores["test_score"]),
    np.std(rf_scores["test_score"])))
print("\tContrastice Self + classifier scores = {} (std.err.={})".format(
    np.mean(contrastive_rf_scores["test_score"]),
    np.std(contrastive_rf_scores["test_score"])))


#model.fit(X)

#clf = SelfSLAE([X.shape[-1]], X.shape[-1], [], [])

#for name, param in clf.named_parameters():
#    if param.requires_grad:
#        print(name, param.data)


#clf(torch.tensor(X))

#model = SimpleAE([128,], X.shape[-1])

#for name, param in model.named_parameters():
#    if param.requires_grad:
#        print(name, param.data)

#output = model(torch.tensor(X))

#print('Output: ', output)