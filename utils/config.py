import logging
import time
import os

from dotenv import load_dotenv


load_dotenv()

DATASETS_FOLDER = os.getenv("datasets_folder", default="datasets/data")
RESULTS_FOLDER = os.getenv("results_folder", default="results/")

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename=f"logs/run-{time.time()}.txt",
    filemode='w',
    level=logging.INFO
)
