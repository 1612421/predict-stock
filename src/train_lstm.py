import math

from src.config import file
from src.config.algo import ALGO_TYPE
from src.predict.LSTM import StockLSTM
from src.predict.ParseDF import parse_roc_df, parseCloseDF

close_df = parseCloseDF(
    file.TRAIN_FILE
) if ALGO_TYPE == "close" else parse_roc_df(file.TRAIN_FILE)
model_name = "lstm_close.h5" if ALGO_TYPE == "close" else "lstm_roc.h5"
target_col = "Close" if ALGO_TYPE == "close" else "ROC"

training_data_len = math.ceil(len(close_df) * .8)
# training_data_len = 987

stockLSTM = StockLSTM()
stockLSTM.trainModel(close_df, training_data_len, model_name, target_col)
