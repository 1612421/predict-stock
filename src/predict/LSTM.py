import pandas as pd
from keras.models import load_model, Sequential
from keras.layers import LSTM, Dense
from pandas.core.tools.datetimes import Scalar
from sklearn.preprocessing import MinMaxScaler
import numpy as np

CLOSE_LSTM_MODEL_NAME = 'saved_lstm_model_close_price.h5'


class StockLSTM:

    def __init__(self):
        self.model_dir = '../models'

    def predict(self, df_predict, model_name):
        test_data = df_predict.values
        scaler = MinMaxScaler(feature_range=(0, 1))
        test_data = scaler.fit_transform(test_data)
        X_test = []

        if len(test_data) == 60:
            X_test.append(test_data)
        else:
            for i in range(60, len(test_data) + 1):
                X_test.append(test_data[i - 60:i, 0])

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        model = load_model(self.model_dir + '/' + model_name)
        closing_price = model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)

        return closing_price

    def trainModel(self, df, training_data_len, saved_model_name):
        df.drop('Date', axis=1, inplace=True)
        final_dataset = df.values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(final_dataset)
        train_data = scaled_data[0:training_data_len, :]
        x_train_data, y_train_data = [], []

        for i in range(60, len(train_data)):
            x_train_data.append(train_data[i - 60:i, 0])
            y_train_data.append(train_data[i, 0])

        x_train_data, y_train_data = np.array(x_train_data), np.array(
            y_train_data)

        x_train_data = np.reshape(
            x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

        lstm_model = Sequential()
        lstm_model.add(
            LSTM(units=50,
                 return_sequences=True,
                 input_shape=(x_train_data.shape[1], 1)))
        lstm_model.add(LSTM(units=50))
        lstm_model.add(Dense(1))

        lstm_model.compile(loss='mean_squared_error', optimizer='adam')
        lstm_model.fit(x_train_data,
                       y_train_data,
                       epochs=1,
                       batch_size=1,
                       verbose=2)

        lstm_model.save(self.model_dir + '/' + saved_model_name)

    # def predict_test(self, data_predict, model_name):
    #     scaler = MinMaxScaler(feature_range=(0, 1))
    #     last_60_days_scaled = scaler.fit_transform(data_predict.values)
    #     x_test = []
    #     x_test.append(last_60_days_scaled)
    #     x_test = np.array(x_test)
    #     x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    #     model = load_model(self.model_dir + '/' + model_name)
    #     predict_val = model.predict(x_test)
    #     predict_val = scaler.inverse_transform(predict_val)

    #     print(predict_val)

    #     return predict_val
