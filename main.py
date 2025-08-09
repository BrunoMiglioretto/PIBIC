import numpy as np
import pandas as pd

from aeon.transformations.collection.convolution_based import Rocket, MiniRocket
from aeon.datasets.tsc_datasets import multivariate
from sklearn.linear_model import RidgeClassifierCV

from utils import config
from utils.config import DATASETS_FOLDER, logger
from utils.utils import transform_series, dimensions_fusion, load_dataset, PAA
#%% md
# #### Configurações
#%%
RESULTS_FILENAME = f'results_final.csv'

reps = ['RP', 'MTF', 'GASF', 'GADF', 'FIRTS', 'CWT']
operations = ["sum", "subtraction", "dot_product", "element_wise"]

#%%
try:
    df_results = pd.read_csv(f"{config.RESULTS_FOLDER}/{RESULTS_FILENAME}")
except FileNotFoundError:
    df_results = pd.DataFrame(columns=[
        "dataset",
        "representation",
        "operation",
        "accuracy",
        "convolution_algorithm",
        "classification_algorithm",
    ])
#%% md
# #### Gerando resultados com apenas o classficador Ridge sem nenhuma transformação ou convolução
#%%
from utils.utils import znorm

for dataset_name in multivariate:
    if df_results[
        (df_results["dataset"] == dataset_name)
        & (df_results["representation"].isnull())
        & (df_results["operation"].isnull())
    ].shape[0] == 1:
        logger.info(f"Dataset {dataset_name} já processado.")
        continue

    try:
        dataset = load_dataset(dataset_name, DATASETS_FOLDER)
        X_train = dataset["X_train"]
        y_train = dataset["y_train"]
        X_test = dataset["X_test"]
        y_test = dataset["y_test"]

        try:
            X_train_transformed = np.array([np.sum([znorm(series) for series in exemple], axis=0) for exemple in X_train])
            X_test_transformed = np.array([np.sum([znorm(series) for series in exemple], axis=0) for exemple in X_test])
           
            algorithm = MiniRocket(n_kernels=10000, n_jobs=-1, random_state=6)
            algorithm.fit(X_train_transformed)
            
            X_train_transformed = algorithm.transform(X_train_transformed)
            X_test_transformed = algorithm.transform(X_test_transformed)
            
            classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            classifier.fit(X_train_transformed, y_train)
            
            accuracy = classifier.score(X_test_transformed, y_test)

            new_result_line = {
                "dataset": dataset_name,
                "representation": None,
                "operation": None,
                "accuracy": accuracy,
                "convolution_algorithm": "MiniRocket",
                "classification_algorithm": "Ridge",
            }
            df_results.loc[len(df_results)] = new_result_line
            df_results.to_csv(f"{config.RESULTS_FOLDER}/{RESULTS_FILENAME}", index=False)

            logger.info("Processamento finalizado com sucesso.")
            
            algorithm = Rocket(n_kernels=10000, n_jobs=-1, random_state=6)
            algorithm.fit(X_train_transformed)
            
            X_train_transformed = algorithm.transform(X_train_transformed)
            X_test_transformed = algorithm.transform(X_test_transformed)
            
            classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            classifier.fit(X_train_transformed, y_train)
            
            accuracy = classifier.score(X_test_transformed, y_test)

            new_result_line = {
                "dataset": dataset_name,
                "representation": None,
                "operation": None,
                "accuracy": accuracy,
                "convolution_algorithm": "Rocket",
                "classification_algorithm": "Ridge",
            }
            df_results.loc[len(df_results)] = new_result_line
            df_results.to_csv(f"{config.RESULTS_FOLDER}/{RESULTS_FILENAME}", index=False)
            
            logger.info("Processamento finalizado com sucesso.")
        except Exception as e:
            logger.error(f"Problema com o dataset {dataset_name}: {e}")
    except Exception as e:
        logger.error(f"Problema ao carregar dataset {dataset_name}: {e}")

#%% md
# - Aplicar Up sampling
# - Down sampling
# - Aplicar o PCA
# - Boxplot com apenas uma categoria
# - Pegar a média dos visinhos quando for nan
# 
# - Mostrar gráficos com
#%% md
# #### Gerando resultados com o classficador Ridge, transformações e convoluções
#%%
for dataset_name in []:
    try:
        if df_results[
            (df_results["dataset"] == dataset_name)
            & ~(df_results["representation"].isnull())
        ].shape[0] == len(reps) * len(operations) * 3: # Teste sem convolução, com Rocket e com MiniRocket
            logger.info(f"Dataset {dataset_name} já processado.")
            continue

        dataset = load_dataset(dataset_name, config.DATASETS_FOLDER)
        X_train = dataset["X_train"]
        y_train = dataset["y_train"]
        X_test = dataset["X_test"]
        y_test = dataset["y_test"]
        
        for representation in reps:
            if df_results[
                (df_results["dataset"] == dataset_name)
                & (df_results["representation"] == representation)
            ].shape[0] == len(operations) * 3: # Teste sem convolução, com Rocket e com MiniRocket 
                logger.info(f"Dataset {dataset_name} com representação {representation} já processado.")
                continue
            
            logger.info(f"Iniciando o processo de transformação das dimensões na representação {representation}")

            transformed_train_series = []
            for exemple in X_train:
                exemple_processed = []
                for series in exemple:
                    if len(series) > 300:
                        series = PAA(series, 300) 
                    t = transform_series(series, representation)
                    exemple_processed.append(t)
                transformed_train_series.append(exemple_processed)
            transformed_test_series = []
            for exemple in X_test:
                exemple_processed= []
                for series in exemple:
                    if len(series) > 300:
                        series = PAA(series, 300)
                    t = transform_series(series, representation)
                    exemple_processed.append(t)
                transformed_test_series.append(exemple_processed)

            logger.info("Finalizado processo de transformação das dimensões com sucesso")

            for operation in operations:
                if df_results[
                    (df_results["dataset"] == dataset_name)
                    & (df_results["representation"] == representation)
                    & (df_results["operation"] == operation)
                ].shape[0] == 3: # Teste sem convolução, com Rocket e com MiniRocket 
                    logger.info(f"Dataset {dataset_name}, representação {representation} e operação {operation} todos as variações já processadas.")
                    continue
                
                logger.info(f"Iniciando processo de fusão das dimensões na operação {operation}")
                X_train_transformed = dimensions_fusion(transformed_train_series, operation)
                X_test_transformed = dimensions_fusion(transformed_test_series, operation)
                
                logger.info("Finalizado processo de fusão")

                try:
                    if df_results[
                        (df_results["dataset"] == dataset_name)
                        & (df_results["representation"] == representation)
                        & (df_results["operation"] == operation)
                        & (df_results["convolution_algorithm"].isnull())
                    ].shape[0] == 0:
                        logger.info("Iniciando processo de treinamento apenas com o classificador Ridge")
                        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
                        classifier.fit(X_train_transformed, y_train)

                        accuracy = classifier.score(X_test_transformed, y_test)

                        new_result_line = {
                            "dataset": dataset_name,
                            "representation": representation,
                            "operation": operation,
                            "accuracy": accuracy,
                            "convolution_algorithm": None,
                            "classification_algorithm": "Ridge",
                        }
                        df_results.loc[len(df_results)] = new_result_line
                        df_results.to_csv(f"{config.RESULTS_FOLDER}/{RESULTS_FILENAME}", index=False)
                    else:
                        logger.info(f"Dataset {dataset_name} com representação {representation}, operação {operation} e sem convolução já processado.")
                except Exception as e:
                    logger.error(f"Problema com o dataset {dataset_name} com o classificador Ridge: {e}")

                try:
                    if df_results[
                        (df_results["dataset"] == dataset_name)
                        & (df_results["representation"] == representation)
                        & (df_results["operation"] == operation)
                        & (df_results["convolution_algorithm"] == "Rocket")
                    ].shape[0] == 0:
                        logger.info("Iniciando processo de treinamento com o classificador Ridge e convolução Rocket")

                        algorithm = Rocket(n_kernels=10000, n_jobs=-1, random_state=6)
                        algorithm.fit(X_train_transformed)

                        X_train_transformed = algorithm.transform(X_train_transformed)
                        X_test_transformed = algorithm.transform(X_test_transformed)

                        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
                        classifier.fit(X_train_transformed, y_train)

                        accuracy = classifier.score(X_test_transformed, y_test)

                        new_result_line = {
                            "dataset": dataset_name,
                            "representation": representation,
                            "operation": operation,
                            "accuracy": accuracy,
                            "convolution_algorithm": "Rocket",
                            "classification_algorithm": "Ridge",
                        }
                        df_results.loc[len(df_results)] = new_result_line
                        df_results.to_csv(f"{config.RESULTS_FOLDER}/{RESULTS_FILENAME}", index=False)
                    else:
                        logger.info(f"Dataset {dataset_name} com representação {representation}, operação {operation} e com convolução Rocket já processado.")
                except Exception as e:
                    logger.error(f"Problema com o dataset {dataset_name} usando Rocket: {e}")

                try:
                    if df_results[
                        (df_results["dataset"] == dataset_name)
                        & (df_results["representation"] == representation)
                        & (df_results["operation"] == operation)
                        & (df_results["convolution_algorithm"] == "MiniRocket")
                    ].shape[0] == 0:
                        logger.info("Iniciando processo de treinamento com o classificador Ridge e convolução MiniRocket")

                        algorithm = MiniRocket(n_kernels=10000, n_jobs=-1, random_state=6)
                        algorithm.fit(X_train_transformed)

                        X_train_transformed = algorithm.transform(X_train_transformed)
                        X_test_transformed = algorithm.transform(X_test_transformed)

                        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
                        classifier.fit(X_train_transformed, y_train)

                        accuracy = classifier.score(X_test_transformed, y_test)

                        new_result_line = {
                            "dataset": dataset_name,
                            "representation": representation,
                            "operation": operation,
                            "accuracy": accuracy,
                            "convolution_algorithm": "MiniRocket",
                            "classification_algorithm": "Ridge",
                        }
                        df_results.loc[len(df_results)] = new_result_line
                        df_results.to_csv(f"{config.RESULTS_FOLDER}/{RESULTS_FILENAME}", index=False)
                    else:
                        logger.info(f"Dataset {dataset_name} com representação {representation}, operação {operation} e com convolução MiniRocket já processado.")
                except Exception as e:
                    logger.error(f"Problema com o dataset {dataset_name} usando MiniRocket: {e}")

        logger.info(f"Finalizado o processamento do dataset {dataset_name}.")
    except Exception as e:
        logger.error(f"Problema ao carregar dataset {dataset_name}: {e}")
    
logger.info("Finalizado o processamento de todos os datasets.")
