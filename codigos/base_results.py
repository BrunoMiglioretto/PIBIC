#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time

import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeClassifierCV
from aeon.transformations.collection.convolution_based import Rocket, MiniRocket
from aeon.datasets.tsc_datasets import multivariate

import config
from config import logger
from utils import load_dataset


# In[2]:


result = {
    "dataset": [],
    "Rocket": [],
    "MiniRocket": [],
}

for dataset_name in multivariate:
    r = {
        "dataset": dataset_name,
        "Rocket": None,
        "MiniRocket": None,
    }
    try:
        dataset = load_dataset(dataset_name, config.DATASETS_FOLDER)
        X_train = dataset["X_train"]
        y_train = dataset["y_train"]
        X_test = dataset["X_test"]
        y_test = dataset["y_test"]
        
        for algorithm_name, Algorithm in [("Rocket", Rocket), ("MiniRocket", MiniRocket)]:
            algorithm = Algorithm(n_kernels=10000, n_jobs=-1, random_state=6)
            algorithm.fit(X_train)
            
            X_train = algorithm.transform(X_train)
            X_test = algorithm.transform(X_test)
            
            classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            classifier.fit(X_train, y_train)
            
            acc = classifier.score(X_test, y_test)
            
            r[algorithm_name] = acc
    except Exception as e:
        logger.error(e)
        logger.error(f"{dataset_name} - {algorithm_name}")

    result["dataset"].append(r["dataset"])
    result["Rocket"].append(r["Rocket"])
    result["MiniRocket"].append(r["MiniRocket"])


# In[4]:


df = pd.DataFrame(result)
df.to_csv(f"{config.RESULTS_FOLDER}/base_results_{time.time()}.csv")

