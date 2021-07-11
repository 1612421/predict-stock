from ast import dump
from os import path
import math

import pandas as pd
from config.algo import ALGO_TYPE
from config.file import PREDICT_DATA_FILE, MODEL_DIR, TRAIN_FILE

from .LSTM import StockLSTM
from .XGBoost import StockXGBoost
from .ParseDF import parseCloseDF, parse_roc_df


class StockManager:
    def __init__(self):
        self.__load_nse_lstm()
        self.__load_nse_xgboost()

        self.lstm_facebook_chart = self.__load_predict_lstm('FB')
        self.xgboost_facebook_chart = self.__load_predict_xgboost('FB')

        self.lstm_tesla_chart = self.__load_predict_lstm('TSLA')
        self.xgboost_tesla_chart = self.__load_predict_xgboost('TSLA')

        self.lstm_apple_chart = self.__load_predict_lstm('AAPL')
        self.xgboost_apple_chart = self.__load_predict_xgboost('AAPL')

        self.lstm_microsoft_chart = self.__load_predict_lstm('MSFT')
        self.xgboost_microsoft_chart = self.__load_predict_xgboost('MSFT')

    def __load_nse_lstm(self):
        stock_lstm = StockLSTM()
        df = parse_roc_df(TRAIN_FILE)
        training_data_len = math.ceil(len(df) * .8)
        close_model_name = 'lstm_close.h5'
        roc_model_name = 'lstm_roc.h5'
        train = df[:training_data_len]
        valid = df[training_data_len:]
        predict_df = df[(training_data_len - 60):-1]
        valid['Close Predict'] = stock_lstm.predict(
            predict_df, path.join(MODEL_DIR, close_model_name), 'Close'
        )
        valid['ROC Predict'] = stock_lstm.predict(
            predict_df, path.join(MODEL_DIR, roc_model_name), 'ROC'
        )

        self.lstm_nse_chart = {'real': train, 'predict': valid}

        # close_df = parseCloseDF(TRAIN_FILE)\
        #     if ALGO_TYPE == "close"\
        #     else parse_roc_df(TRAIN_FILE)
        # model_name = "lstm_close.h5"\
        #     if ALGO_TYPE == "close"\
        #     else "lstm_roc.h5"
        # target_col = 'Close'\
        #     if ALGO_TYPE == "close"\
        #     else "ROC"

        # train = close_df[:987]
        # valid = close_df[987:]
        # close_predict_df = close_df[(987 - 60):-1]

        # if (ALGO_TYPE == "close"):
        #     valid['Predict'] = stock_lstm.predict(
        #         close_predict_df, path.join(MODEL_DIR, model_name), target_col
        #     )
        # elif (ALGO_TYPE == "roc"):
        #     valid['PredictROC'] = stock_lstm.predict(
        #         close_predict_df, path.join(MODEL_DIR, model_name), target_col
        #     )
        #     valid = valid.assign(Predict=0)
        #     valid['Predict'][0] = valid['Close'][0]
        #     for i in range(1, len(valid)):
        #         valid['Predict'][i] = valid['Predict'][
        #             i - 1] * (valid['PredictROC'][i] + 1)

        # self.nse_close_chart = {'real': train, 'predict': valid}

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

    def __load_nse_xgboost(self):
        stock_xgboost = StockXGBoost()
        df = parse_roc_df(TRAIN_FILE)
        training_data_len = math.ceil(len(df) * .8)
        close_model_name = 'xgboost_close.json'
        roc_model_name = 'xgboost_roc.json'
        train = df[:training_data_len]
        valid = df[training_data_len:]
        predict_df = df[(training_data_len - 1):-1]
        valid['Close Predict'] = stock_xgboost.predict(
            predict_df, path.join(MODEL_DIR, close_model_name), 'Close'
        )
        valid['ROC Predict'] = stock_xgboost.predict(
            predict_df, path.join(MODEL_DIR, roc_model_name), 'ROC'
        )

        self.xgboost_nse_chart = {'real': train, 'predict': valid}

    def __load_predict_lstm(self, filter_brand):
        stock_lstm = StockLSTM()
        df = parse_roc_df(PREDICT_DATA_FILE, filter_brand)
        training_data_len = math.ceil(len(df) * .8)
        close_model_name = 'lstm_close.h5'
        roc_model_name = 'lstm_roc.h5'
        train = df[:training_data_len]
        valid = df[training_data_len:]
        predict_df = df[(training_data_len - 60):-1]
        valid['Close Predict'] = stock_lstm.predict(
            predict_df, path.join(MODEL_DIR, close_model_name), 'Close'
        )
        valid['ROC Predict'] = stock_lstm.predict(
            predict_df, path.join(MODEL_DIR, roc_model_name), 'ROC'
        )

        return {'real': train, 'predict': valid}

    def __load_predict_xgboost(self, filter_brand):
        stock_xgboost = StockXGBoost()
        df = parse_roc_df(PREDICT_DATA_FILE, filter_brand)
        training_data_len = math.ceil(len(df) * .8)
        close_model_name = 'xgboost_close.json'
        roc_model_name = 'xgboost_roc.json'
        train = df[:training_data_len]
        valid = df[training_data_len:]
        predict_df = df[(training_data_len - 1):-1]
        valid['Close Predict'] = stock_xgboost.predict(
            predict_df, path.join(MODEL_DIR, close_model_name), 'Close'
        )
        valid['ROC Predict'] = stock_xgboost.predict(
            predict_df, path.join(MODEL_DIR, roc_model_name), 'ROC'
        )

        return {'real': train, 'predict': valid}