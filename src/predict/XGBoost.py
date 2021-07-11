from .LSTM import StockLSTM
from config import file
from pandas.core.frame import DataFrame
from os import path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor, Booster, DMatrix


class StockXGBoost(StockLSTM):
    def predict(
        self, df_predict: DataFrame, model_file: str, target_col: str
    ) -> DataFrame:
        df = df_predict[[target_col]].copy()
        x_test = DMatrix(df.values)
        model = Booster()
        model.load_model(model_file)
        predictions = model.predict(x_test)
        predictions = np.array(predictions).reshape(-1, 1)
        return predictions

    def trainModel(
        self, df: DataFrame, training_data_len: int, saved_model_name: str,
        target_col: str
    ):
        df = df[:training_data_len]
        train_df = df[[target_col]].copy()
        train_df['target'] = train_df[target_col].shift(-1)
        train_df = train_df.values[:-1]
        x_train = train_df[:, :-1]
        y_train = train_df[:, -1]
        model = XGBRegressor(objective="reg:squarederror", n_estimators=1000)
        model.fit(x_train, y_train)
        model.save_model(path.join(file.MODEL_DIR, saved_model_name))
