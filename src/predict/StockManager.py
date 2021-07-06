from os import path

import pandas as pd
from config.file import PREDICT_DATA_FILE, MODEL_DIR, TRAIN_FILE

from .LSTM import StockLSTM
from .ParseDF import parseCloseDF


class StockManager:
    def __init__(self):
        self.__load_nse_lstm()

    def __load_nse_lstm(self):
        stock_lstm = StockLSTM()
        close_df = parseCloseDF(TRAIN_FILE)
        train = close_df[:987]
        valid = close_df[987:]
        close_predict_df = close_df[(987 - 60):-1]
        valid['Predict'] = stock_lstm.predict(
            close_predict_df, path.join(MODEL_DIR, "lstm_close.h5")
        )

        self.nse_close_chart = {'real': train, 'predict': valid}

        # Predict next day close price of Apple base on 60 days before
        df = pd.read_csv(PREDICT_DATA_FILE)
        apple_data = df[df['Stock'] == 'AAPL']
        apple_data['Date'] = pd.to_datetime(apple_data.Date, format='%Y-%m-%d')
        new_df = apple_data.filter(['Close'])
        last_60_days = new_df[-60:]
        temp = stock_lstm.predict(
            last_60_days, path.join(MODEL_DIR, "lstm_close.h5")
        )
        print(temp)
