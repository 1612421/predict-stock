from ast import dump
from os import path

import pandas as pd
from config.algo import ALGO_TYPE
from config.file import PREDICT_DATA_FILE, MODEL_DIR, TRAIN_FILE

from .LSTM import StockLSTM
from .ParseDF import parseCloseDF, parse_roc_df


class StockManager:
    def __init__(self):
        self.__load_nse_lstm()

    def __load_nse_lstm(self):
        stock_lstm = StockLSTM()

        close_df = parseCloseDF(TRAIN_FILE)\
            if ALGO_TYPE == "close"\
            else parse_roc_df(TRAIN_FILE)
        model_name = "lstm_close.h5"\
            if ALGO_TYPE == "close"\
            else "lstm_roc.h5"
        target_col = 'Close'\
            if ALGO_TYPE == "close"\
            else "ROC"

        train = close_df[:987]
        valid = close_df[987:]
        close_predict_df = close_df[(987 - 60):-1]

        if (ALGO_TYPE == "close"):
            valid['Predict'] = stock_lstm.predict(
                close_predict_df, path.join(MODEL_DIR, model_name), target_col
            )
        elif (ALGO_TYPE == "roc"):
            valid['PredictROC'] = stock_lstm.predict(
                close_predict_df, path.join(MODEL_DIR, model_name), target_col
            )
            valid = valid.assign(Predict=0)
            valid['Predict'][0] = valid['Close'][0]
            for i in range(1, len(valid)):
                valid['Predict'][i] = valid['Predict'][
                    i - 1] * (valid['PredictROC'][i] + 1)

        self.nse_close_chart = {'real': train, 'predict': valid}

        # Predict next day close price of Apple base on 60 days before
        # df = pd.read_csv(PREDICT_DATA_FILE)
        # apple_data = df[df['Stock'] == 'AAPL']
        # apple_data['Date'] = pd.to_datetime(apple_data.Date, format='%Y-%m-%d')
        # new_df = apple_data.filter(['Close'])
        # last_60_days = new_df[-60:]
        # temp = stock_lstm.predict(
        #     last_60_days, path.join(MODEL_DIR, model_name), target_col
        # )
        # print(temp)
