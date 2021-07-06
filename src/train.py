import math
from predict.LSTM import StockLSTM, CLOSE_LSTM_MODEL_NAME
from predict.ParseDF import parseCloseDF

close_df = parseCloseDF('../dataset/NSE-Tata.csv')
training_data_len = math.ceil(len(close_df) * .8)
training_data_len = 987
stockLSTM = StockLSTM('../dataset/NSE-Tata.csv')
stockLSTM.trainModel(close_df, training_data_len, CLOSE_LSTM_MODEL_NAME)
