import openml
import os
from tqdm import tqdm
import numpy as np
import sys

open_ml_api_key = sys.argv[1]

os.makedirs("../data",exist_ok=True)
os.makedirs("../data/open_ml",exist_ok=True)

openml.config.apikey = open_ml_api_key  # set the OpenML Api Key

# as shown in https://hal.archives-ouvertes.fr/hal-03723551

SUITE_IDS = [
    #297, # Regression on numerical features
    298, # Classification on numerical features
    #299, # Regression on numerical and categorical features
    304 # Classification on numerical and categorical features
]
for SUITE_ID in SUITE_IDS:
    benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite
    for task_id in tqdm(benchmark_suite.tasks):  # iterate over all tasks
        task = openml.tasks.get_task(task_id)  # download the OpenML task
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        X.insert(loc=0, column='class', value=y)
        X.to_csv("data/open_ml/{}.csv".format(dataset.name),index=False)
