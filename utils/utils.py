from aeon.datasets import load_classification
from aeon.datasets import load_from_ts_file

from utils.config import logger


def load_dataset(dataset_name, dataset_folder):
    try:
        logger.info(f"Carregando {dataset_name}")

        X_train, y_train = load_from_ts_file(f"{dataset_folder}/{dataset_name}/{dataset_name}_TRAIN.ts")
        X_test, y_test = load_from_ts_file(f"{dataset_folder}/{dataset_name}/{dataset_name}_TEST.ts")

        logger.info("Carregamento finalizado com sucesso")

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
    except Exception as e:
        logger.warning(e)
        logger.warning(f"Não foi possível carregar o dataset {dataset_name} armazenados na máquina local")
        logger.info(f"Iniciando download do dataset {dataset_name}")

        X_train, y_train = load_classification(dataset_name, split="Train")
        X_test, y_test = load_classification(dataset_name, split="Test")

        logger.info("Download finalizado com sucesso")

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
