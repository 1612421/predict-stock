import math

from config import file
from config.algo import ALGO_TYPE
from predict.XGBoost import StockXGBoost
from predict.ParseDF import parseCloseDF, parse_roc_df

close_df = parseCloseDF(
    file.TRAIN_FILE
) if ALGO_TYPE == "close" else parse_roc_df(file.TRAIN_FILE)
model_name = "xgboost_close.json" if ALGO_TYPE == "close" else "xgboost_roc.json"
target_col = "Close" if ALGO_TYPE == "close" else "ROC"

training_data_len = math.ceil(len(close_df) * .8)
# training_data_len = 987

stockLSTM = StockXGBoost()
stockLSTM.trainModel(close_df, training_data_len, model_name, target_col)
