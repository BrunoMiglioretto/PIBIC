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

    try:
        series = np.array(znorm(series))
        if representation == "CWT":
            try:
                coeffs, freqs = pywt.cwt(series, scales=np.arange(1, len(series) + 1), wavelet='morl') # morl
                im_final = coeffs
            except Exception as e:
                logger.error(f"Problema ao transformar uma série em CWT: {e}")
        elif representation == "MTF":
            try:
                series = series.reshape(1, len(series))
                mtf = MarkovTransitionField(strategy='normal') #n_bins=4, strategy='uniform'
                X_mtf = mtf.fit_transform(series)
                im_final = X_mtf[0]
            except Exception as e:
                logger.error(f"Problema ao transformar uma série em MTF: {e}")
        elif representation == "GADF":
            try:
                series = series.reshape(1, len(series))
                gaf = GramianAngularField(method='difference')
                X_gaf = gaf.fit_transform(series)
                im_final = X_gaf[0]
            except Exception as e:
                logger.error(f"Problema ao transformar uma série em GADF: {e}")
        elif representation == "GASF":
            try:
                series = series.reshape(1, len(series))
                gaf = GramianAngularField(method='summation')
                X_gaf = gaf.fit_transform(series)
                im_final = X_gaf[0]
            except Exception as e:
                logger.error(f"Problema ao transformar uma série em GASF: {e}")
        elif representation == "RP":
            try:
                series = series.reshape(1, len(series))
                rp = RecurrencePlot(threshold='distance')
                X_rp = rp.fit_transform(series)
                im_final = X_rp[0]
            except Exception as e:
                logger.error(f"Problema ao transformar uma série em RP: {e}")
        elif representation == "FIRTS":
            try:
                series = series.reshape(1, len(series))
                mtf = MarkovTransitionField(n_bins=4, strategy='uniform')
                X_mtf = mtf.fit_transform(series)
                gaf = GramianAngularField(method='difference')
                X_gaf = gaf.fit_transform(series)
                rp = RecurrencePlot(threshold='distance')
                X_rp = rp.fit_transform(series)
                im_final = (X_mtf[0] + X_gaf[0] + X_rp[0])
            except Exception as e:
                logger.error(f"Problema ao transformar uma série em FIRTS: {e}")
        return im_final
    except Exception as e:
        logger.error(f"Problema ao transformar uma série com o znorm: {e}")


def dimensions_fusion(img_dataset, operation):
    """
    operation: sum, subtraction, dot_product, element_wise
    """
    try:
        new_data = []
        for dataset in img_dataset:
            imgs = dataset.copy()
            img_final = imgs.pop()
            for img in imgs:
                if operation == 'sum':
                    try:
                        img_final += img
                    except Exception as e:
                        logger.error(f"Problema ao fundir usando sum: {e}")
                elif operation == 'subtraction':
                    try:
                        img_final -= img
                    except Exception as e:
                        logger.error(f"Problema ao fundir usando subtraction: {e}")
                elif operation == 'dot_product':
                    try:
                        img_final = np.dot(img_final, img)
                    except Exception as e:
                        logger.error(f"Problema ao fundir usando dot_product: {e}")
                elif operation == 'element_wise':
                    try:
                        img_final = np.multiply(img_final, img)
                    except Exception as e:
                        logger.error(f"Problema ao fundir usando element_wise: {e}")

            try:
                flatten_img = img_final.flatten()
                new_data.append(flatten_img)

                new_series = np.array(new_data)
            except Exception as e:
                logger.error(f"Problema ao transformar as imagens em um numpy array: {e}")
            return new_series
    except Exception as e:
        logger.error(f"Problema geral ao fundir as séries: {e}")


def PAA(s, w):
    s = np.asarray(s)
    n = len(s)
    res = np.zeros(w)

    for i in range(w):
        start_idx = i * n // w
        end_idx = (i + 1) * n // w
        res[i] = np.mean(s[start_idx:end_idx])

    return res
