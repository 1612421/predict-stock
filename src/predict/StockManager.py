from .LSTM import StockLSTM, CLOSE_LSTM_MODEL_NAME
from .ParseDF import parseCloseDF
import pandas as pd


class StockManager:
    def __init__(self):
        self.__load_nse_lstm()

    def __load_nse_lstm(self):
        stock_lstm = StockLSTM()
        close_df = parseCloseDF('../dataset/NSE-Tata.csv')
        train = close_df[:987]
        valid = close_df[987:]
        close_predict_df = close_df[(987 - 60):-1]
        valid['Predict'] = stock_lstm.predict(
            close_predict_df, CLOSE_LSTM_MODEL_NAME
        )

        self.nse_close_chart = {'real': train, 'predict': valid}

        # Predict next day close price of Apple base on 60 days before
        df = pd.read_csv("../dataset/stock_data.csv")
        apple_data = df[df['Stock'] == 'AAPL']
        apple_data['Date'] = pd.to_datetime(apple_data.Date, format='%Y-%m-%d')
        new_df = apple_data.filter(['Close'])
        last_60_days = new_df[-60:]
        temp = stock_lstm.predict(last_60_days, CLOSE_LSTM_MODEL_NAME)
        print(temp)
