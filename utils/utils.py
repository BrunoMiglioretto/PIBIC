from aeon.datasets import load_classification
from aeon.datasets import load_from_ts_file


import pywt
import numpy as np

from pyts.image import MarkovTransitionField
from pyts.image import GramianAngularField
from pyts.image import RecurrencePlot

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


def znorm(x):
    """
    função para normalizar as séries na mesma escala
    a série ficará com uma média próxima de 0 e desvio-padrão próximo de 1
    """

    x_znorm = (x - np.mean(x)) / np.std(x)
    return x_znorm


def transform_series(series, representation):
    """
    função que transforma uma série de entrada em uma imagem em 2D.
    transformações que serão exploradas: CWT, MTF, GADF, GASF, RP e FIRTS
    referência para entender um pouco melhor: https://pyts.readthedocs.io/en/stable/modules/image.html
    """

    series = np.array(znorm(series))
    if representation == "CWT":
        coeffs, freqs = pywt.cwt(series, scales=np.arange(1, len(series) + 1), wavelet='morl') # morl
        im_final = coeffs
    elif representation == "MTF":
        series = series.reshape(1, len(series))
        mtf = MarkovTransitionField(strategy='normal') #n_bins=4, strategy='uniform'
        X_mtf = mtf.fit_transform(series)
        im_final = X_mtf[0]
    elif representation == "GADF":
        series = series.reshape(1, len(series))
        gaf = GramianAngularField(method='difference')
        X_gaf = gaf.fit_transform(series)
        im_final = X_gaf[0]
    elif representation == "GASF":
        series = series.reshape(1, len(series))
        gaf = GramianAngularField(method='summation')
        X_gaf = gaf.fit_transform(series)
        im_final = X_gaf[0]
    elif representation == "RP":
        series = series.reshape(1, len(series))
        rp = RecurrencePlot(threshold='distance')
        X_rp = rp.fit_transform(series)
        im_final = X_rp[0]
    elif representation == "FIRTS":
        series = series.reshape(1, len(series))
        mtf = MarkovTransitionField(n_bins=4, strategy='uniform')
        X_mtf = mtf.fit_transform(series)
        gaf = GramianAngularField(method='difference')
        X_gaf = gaf.fit_transform(series)
        rp = RecurrencePlot(threshold='distance')
        X_rp = rp.fit_transform(series)
        im_final = (X_mtf[0] + X_gaf[0] + X_rp[0])
    return im_final


def dimensions_fusion(img_dataset, operation):
    """
    operation: sum, subtraction, dot_product, element_wise
    """

    new_data = []
    for dataset in img_dataset:
        imgs = dataset.copy()
        img_final = imgs.pop()
        for img in imgs:
            if operation == 'sum':
                img_final += img
            elif operation == 'subtraction':
                img_final -= img
            elif operation == 'dot_product':
                img_final = np.dot(img_final, img)
            elif operation == 'element_wise':
                img_final = np.multiply(img_final, img)

        flatten_img = img_final.flatten()
        new_data.append(flatten_img)

    return np.array(new_data)


def PAA(s, w):
    s = np.asarray(s)
    n = len(s)
    res = np.zeros(w)

    for i in range(w):
        start_idx = i * n // w
        end_idx = (i + 1) * n // w
        res[i] = np.mean(s[start_idx:end_idx])

    return res
