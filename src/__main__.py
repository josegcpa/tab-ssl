import argparse
import yaml
import time
import json
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import ShuffleSplit
from multiprocessing import Pool

from .data_generator import AutoDataset
from .data import supported_datasets
from .decomposition_utilities import supported_decompositions

supported_learning_algorithms = {
    "rf":RandomForestClassifier,
    "linear":ElasticNetCV
}

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset",required=True,
                        choices=supported_datasets.keys(),
                        help="Dataset ID")
    parser.add_argument("--decomposition",required=True,
                        choices=list(supported_decompositions.keys())+["none"],
                        help="Decomposition algorithm ID")
    parser.add_argument("--learning_algorithm",required=True,
                        choices=["rf","linear"],
                        help="Name of learning algorithm")
    parser.add_argument("--seed",default=42,
                        help="Random seed")
    parser.add_argument("--n_folds",default=5,type=int,
                        help="Number of folds")
    parser.add_argument("--decomposition_config",
                        help="Path to yaml file with decomposition args")
    parser.add_argument("--learning_algorithm_config",
                        help="Path to yaml file with learning algorithm args")
    parser.add_argument("--n_workers",default=0,type=int,
                        help="Number of concurrent processes")

    args,unknown_args = parser.parse_known_args()
    
    if args.decomposition_config is not None:
        decomposition_config = yaml.load(args.decomposition_config)
    else:
        decomposition_config = {}
    if args.decomposition != "ipca":
        decomposition_config["random_state"] = args.seed
    
    if args.learning_algorithm_config is not None:
        learning_algorithm_config = yaml.load(args.learning_algorithm_config)
    else:
        learning_algorithm_config = {}
    
    dataset = supported_datasets[
        args.dataset]
    decomposition = supported_decompositions[
        args.decomposition]
    learning_algorithm = supported_learning_algorithms[
        args.learning_algorithm]

    X,y,cat_cols = dataset()

    pipeline = Pipeline([
        ("auto_dataset",AutoDataset(cat_cols=cat_cols)),
        ("nzv",VarianceThreshold()),
        ("scaler",StandardScaler()),
        ("dec",decomposition(**decomposition_config)),
        ("learner",learning_algorithm(**learning_algorithm_config,
                                      random_state=args.seed))
    ])

    cv = ShuffleSplit(args.n_folds,random_state=args.seed)
    splits = cv.split(X)
    
    def wraper(train_val_idxs):
        train_idxs,val_idxs = train_val_idxs
        train_X = X[train_idxs]
        train_y = y[train_idxs]
        val_X = X[val_idxs]
        val_y = y[val_idxs]

        time_a = time.time()
        pipeline.fit(train_X,train_y)
        time_b = time.time()
        
        elapsed = time_b - time_a
        
        # export only what is strictly necessary 
        # to compute downstream metrics
        pred = pipeline.predict(val_X).tolist()
        pred_proba = pipeline.predict_proba(val_X).tolist()
        output_dict = {
            "pred":pred,
            "pred_proba":pred_proba,
            "y":val_y.tolist(),
            "n_classes":len(np.unique(val_y)),
            "time_elapsed":elapsed
        }
        return output_dict

    if args.n_workers == 0:
        fold_scoring = []
        for i,(train_idxs,val_idxs) in enumerate(splits):
            output_dict = wraper((train_idxs,val_idxs))
            fold_scoring.append(output_dict)
    
    else:
        pool = Pool(args.n_workers)
        fold_scoring = pool.map(wraper,splits)
    
    print(json.dumps(fold_scoring,indent=2))