from predict.LSTM import StockLSTM

stockLSTM = StockLSTM('../dataset/NSE-Tata.csv')
stockLSTM.train_close_price()