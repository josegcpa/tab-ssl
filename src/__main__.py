import argparse
import yaml
import time
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from multiprocessing import Pool

from .data_generator import AutoDataset
from .data import supported_datasets
from .decomposition_utilities import supported_decompositions

from .stdgp.StdGP import StdGP
from .stdgp.save_csv import save_csv

supported_learning_algorithms = {
    "rf":RandomForestClassifier,
    "linear":ElasticNetCV,
    "stdgp": StdGP,
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
                        choices=["rf","linear","stdgp"],
                        help="Name of learning algorithm")
    parser.add_argument("--unsupervised_fraction",type=float,
                        help="Fraction of samples to be used for unsupervised\
                            training.",default=None)
    parser.add_argument("--seed",default=42,
                        help="Random seed")
    parser.add_argument("--n_folds",default=5,type=int,
                        help="Number of folds")
    parser.add_argument("--decomposition_config",
                        help="Path to yaml file with decomposition args")
    parser.add_argument("--learning_algorithm_config",type=str,
                        help="Path to yaml file with learning algorithm args")
    parser.add_argument("--n_workers",default=0,type=int,
                        help="Number of concurrent processes")
    parser.add_argument("--output_file",default=None,
                        help="Path for output json file. If not specified \
                            prints json.")

    args,unknown_args = parser.parse_known_args()
    
    dataset = supported_datasets[
        args.dataset]
    learning_algorithm = supported_learning_algorithms[
        args.learning_algorithm]

    X,y,cat_cols = dataset()
    
    if args.decomposition_config is not None:
        decomposition_config = yaml.load(args.decomposition_config)
    else:
        decomposition_config = {}
    if args.decomposition != "ipca":
        decomposition_config["random_state"] = args.seed
    if args.decomposition in ["ae","vime"]:
        decomposition_config["cat_cols"] = cat_cols


    if args.learning_algorithm_config is not None:
        with open(args.learning_algorithm_config, 'r') as o:
            learning_algorithm_config = yaml.load(o)
    else:
        learning_algorithm_config = {}

    preproc_transforms = [
        ("auto_dataset",AutoDataset(cat_cols=cat_cols)),
        ("nzv",VarianceThreshold()),
        ("scaler",StandardScaler())]

    if args.decomposition != "none":
        decomposition = supported_decompositions[
            args.decomposition]
        preproc_transforms.append(
            ("dec",decomposition(**decomposition_config)))

    pipeline_preprocessing = Pipeline(preproc_transforms)
    learner = None
    if args.learning_algorithm != "stdgp" and args.learning_algorithm != "m3gp":
        learner = learning_algorithm(
            **learning_algorithm_config,random_state=args.seed)
    cv = ShuffleSplit(args.n_folds,random_state=args.seed)
    splits = cv.split(X)
    
    def wraper(train_val_idxs, idx, learner=None):
        train_idxs,val_idxs = train_val_idxs
        train_X = X[train_idxs]
        train_y = y[train_idxs]
        val_X = X[val_idxs]
        val_y = y[val_idxs]

        time_a = time.time()
        if args.unsupervised_fraction is not None:
            train_X,train_X_unsupervised,train_y,_ = train_test_split(
                train_X,train_y,test_size=args.unsupervised_fraction,

                stratify=train_y,random_state=args.seed)
            print("Unsupervised learning array shape:",
                  train_X_unsupervised.shape)
        else:
            train_X_unsupervised = train_X
        print("Supervised learning array shape:",
              train_X.shape)
        pipeline_preprocessing.fit(train_X_unsupervised)
        if args.learning_algorithm != "stdgp" and args.learning_algorithm != "m3gp":
            learner.fit(pipeline_preprocessing.transform(train_X),train_y)
        else:
            # GP requires both training and validation data to get the metrics over time
            tr_data = pipeline_preprocessing.transform(train_X)
            tr_data = pd.DataFrame(tr_data, columns=['X'+str(i) for i in range(tr_data.shape[1])])
            tv_data = pipeline_preprocessing.transform(val_X)
            tv_data = pd.DataFrame(tv_data, columns=['X' + str(i) for i in range(tv_data.shape[1])])

            learner = learning_algorithm(**learning_algorithm_config, random_state=args.seed)
            learner.fit(tr_data, train_y, tv_data, val_y)

            save_csv(learner, args.dataset+'_'+str(args.unsupervised_fraction), idx)
        time_b = time.time()
        
        elapsed = time_b - time_a
        
        # export only what is strictly necessary 
        # to compute downstream metrics
        if args.learning_algorithm != "stdgp" and args.learning_algorithm != "m3gp":
            transformed_X = pipeline_preprocessing.transform(val_X)
            pred = learner.predict(transformed_X).tolist()
        else:
            # Test data already transformed for GP above
            #tv_data = pipeline_preprocessing.transform(val_X)
            #tv_data = pd.DataFrame(tv_data, columns=['X'+str(i) for i in range(tv_data.shape[1])])
            pred = learner.getBestIndividual().predict(tv_data)
        try:
            pred_proba = learner.predict_proba(transformed_X).tolist()
        except:
            pred_proba = None
        nc = len(np.unique(val_y))
        f1 = f1_score(val_y,pred,average="binary" if nc==2 else "micro")
        print("Fold concluded\n\tF1-score={}".format(f1))
        output_dict = {
            "pred":pred,
            "pred_proba":pred_proba,
            "y":val_y.tolist(),
            "n_classes":nc,
            "f1-score":f1,
            "time_elapsed":elapsed
        }
        return output_dict

    if args.n_workers == 0:
        fold_scoring = []
        for i,(train_idxs,val_idxs) in enumerate(splits):
            output_dict = wraper((train_idxs,val_idxs), i, learner)
            fold_scoring.append(output_dict)

    # Needs fix
    else:
        pool = Pool(args.n_workers)
        fold_scoring = pool.map(wraper,splits)
        
    out = json.dumps(fold_scoring,indent=2)
    if args.output_file is not None:
        with open(args.output_file,"w") as o:
            o.write(out)
    else:
        print(out)