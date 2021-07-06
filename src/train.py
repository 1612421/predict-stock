import math
from os import path

from config import file
from predict.LSTM import StockLSTM
from predict.ParseDF import parseCloseDF

close_df = parseCloseDF(file.TRAIN_FILE)
training_data_len = math.ceil(len(close_df) * .8)
training_data_len = 987

stockLSTM = StockLSTM()
stockLSTM.trainModel(close_df, training_data_len, "lstm_close.h5")
