import json
import logging
import time

CONFIG_FILE = "/Users/brunomiglioretto/puc/iniciacao/pibic/config.json"

RESULTS_FILENAME = f'results_final.csv'

representations = ['RP', 'MTF', 'GASF', 'GADF', 'FIRTS', 'CWT']
operations = ["sum", "subtraction", "dot_product", "element_wise"]


with open(CONFIG_FILE) as f:
    content = f.read()
    config = json.loads(content)

DATASETS_FOLDER = config.get("datasets_folder")
RESULTS_FOLDER = config.get("results_folder")
LOGS_FOLDER = config.get("logs_folder")
SKIP_DATASETS = config.get("skip_datasets")


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename=f"{LOGS_FOLDER}/run-{time.time()}.txt",
    filemode='w',
    level=logging.INFO
)
