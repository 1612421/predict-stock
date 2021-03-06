import math
from os import path
from pandas.core.frame import DataFrame
from datetime import datetime, timedelta

from numpy import reshape
from pandas.core.generic import NDFrame
from pandas.core.series import Series
from src.config.file import MODEL_DIR, PREDICT_DATA_FILE, TRAIN_FILE

from .LSTM import StockLSTM
from .ParseDF import parse_roc_df
from .XGBoost import StockXGBoost


class StockManager:
    is_loaded = False
    is_future_predicted = False

    predicted_chart: dict[str, dict[str, NDFrame]] = dict(
        {
            'lstm': dict(),
            'xgboost': dict()
        }
    )
    predicted_future_chart: dict[str, dict[str, NDFrame]] = dict(
        {
            "lstm": dict(),
            "xgboost": dict()
        }
    )

    def load_all(self):
        self.__load_nse_lstm()
        self.__load_nse_xgboost()

        for brand in ['FB', 'TSLA', 'AAPL', 'MSFT']:
            self.predicted_chart['lstm'][brand] = self.__load_predict_lstm(
                brand
            )
            self.predicted_chart['xgboost'][
                brand] = self.__load_predict_xgboost(brand)

        self.is_loaded = True

    def load_future_predict(self):
        if (not self.is_loaded):
            raise RuntimeError("Need data loaded first")
        for brand in ['FB', 'TSLA', 'AAPL', 'MSFT']:
            self.predicted_future_chart['lstm'][
                brand] = self.__load_future_predict('lstm', brand)
            self.predicted_future_chart['xgboost'][
                brand] = self.__load_future_predict('xgboost', brand)
        self.is_future_predicted = True

    def __load_future_predict(self, method_name: str, brand: str):
        df = parse_roc_df(PREDICT_DATA_FILE, brand)
        method = StockLSTM()
        close_model_name = 'lstm_close.h5'
        roc_model_name = 'lstm_roc.h5'

        if (method_name == 'xgboost'):
            method = StockXGBoost()
            close_model_name = 'xgboost_close.json'
            roc_model_name = 'xgboost_roc.json'

        valid = df.copy()
        valid.insert(0, 'Close Predict', value=None)
        valid.insert(0, 'ROC Predict', value=None)
        predict_df = df.copy()
        predict_date = df.index[-1]

        for _ in range(7):
            predict_date += timedelta(days=1)
            close_predict = method.predict(
                predict_df[-60:], path.join(MODEL_DIR, close_model_name),
                'Close'
            )
            close_predict = reshape(close_predict, newshape=(1, -1))
            # print (close_predict)
            # continue
            adding = DataFrame(
                index=[predict_date],
                data=close_predict[0][0],
                columns=['Close', 'Close Predict']
            )
            valid = valid.append(adding)
            predict_df = predict_df.append(adding)

        predict_df = df.copy()
        predict_date = df.index[-1]
        for _ in range(7):
            predict_date += timedelta(days=1)
            roc_predict = method.predict(
                predict_df[-60:], path.join(MODEL_DIR, roc_model_name), 'ROC'
            )
            roc_predict = reshape(roc_predict, newshape=(1, -1))
            # print (roc_predict)
            # continue
            adding = DataFrame(
                index=[predict_date],
                data=roc_predict[0][0],
                columns=['Close', 'ROC Predict']
            )
            valid = valid.append(adding)
            predict_df = predict_df.append(adding)

        # print(valid[-60:])

        return valid

    def __load_nse_lstm(self):
        stock_lstm = StockLSTM()
        df = parse_roc_df(TRAIN_FILE)
        training_data_len = math.ceil(len(df) * .8)
        close_model_name = 'lstm_close.h5'
        roc_model_name = 'lstm_roc.h5'
        train = df[:training_data_len]
        valid = df[training_data_len:]
        predict_df = df[(training_data_len - 60):-1]

        valid.insert(
            0, 'Close Predict',
            stock_lstm.predict(
                predict_df, path.join(MODEL_DIR, close_model_name), 'Close'
            )
        )
        valid.insert(
            0, 'ROC Predict',
            stock_lstm.predict(
                predict_df, path.join(MODEL_DIR, roc_model_name), 'ROC'
            )
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
        train = df.iloc[:training_data_len]
        valid = df.iloc[training_data_len:]
        predict_df = df.iloc[(training_data_len - 1):-1]
        valid.insert(
            0, 'Close Predict',
            stock_xgboost.predict(
                predict_df, path.join(MODEL_DIR, close_model_name), 'Close'
            )
        )
        valid.insert(
            0, 'ROC Predict',
            stock_xgboost.predict(
                predict_df, path.join(MODEL_DIR, roc_model_name), 'ROC'
            )
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
        valid.insert(
            0, 'Close Predict',
            stock_lstm.predict(
                predict_df, path.join(MODEL_DIR, close_model_name), 'Close'
            )
        )
        valid.insert(
            0, 'ROC Predict',
            stock_lstm.predict(
                predict_df, path.join(MODEL_DIR, roc_model_name), 'ROC'
            )
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
        valid.insert(
            0, 'Close Predict',
            stock_xgboost.predict(
                predict_df, path.join(MODEL_DIR, close_model_name), 'Close'
            )
        )
        valid.insert(
            0, 'ROC Predict',
            stock_xgboost.predict(
                predict_df, path.join(MODEL_DIR, roc_model_name), 'ROC'
            )
        )

        return {'real': train, 'predict': valid}
