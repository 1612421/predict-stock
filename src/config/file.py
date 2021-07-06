from os import path

TRAIN_FILE = path.realpath(
    path.join(path.dirname(__file__), "../..", "dataset", "NSE-Tata.csv")
)
PREDICT_DATA_FILE = path.realpath(
    path.join(path.dirname(__file__), "../..", "dataset", "stock_data.csv")
)

MODEL_DIR = path.realpath(path.join(path.dirname(__file__), "../..", "models"))
