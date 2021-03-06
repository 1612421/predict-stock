import pandas as pd
from pandas.core.frame import DataFrame


def parseCloseDF(data_file_path: str) -> DataFrame:
    df = pd.read_csv(data_file_path)
    df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df.index = df['Date']
    data = df.sort_index(ascending=True, axis=0)
    new_dataset = pd.DataFrame(
        index=range(0, len(df)), columns=['Date', 'Close']
    )

    for i in range(0, len(data)):
        new_dataset['Date'][i] = data['Date'][i]
        new_dataset['Close'][i] = data['Close'][i]

    new_dataset.index = new_dataset.Date
    new_dataset.drop('Date', axis=1, inplace=True)

    return new_dataset


def parse_roc_df(data_file_path: str, filter_branch=None) -> DataFrame:
    df = pd.read_csv(data_file_path)

    if filter_branch:
        df = df[df['Stock'] == filter_branch]
    df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df.index = df['Date']
    data = df.sort_index(ascending=True, axis=0)
    new_dataset = pd.DataFrame(
        index=range(0, len(df)), columns=['Date', 'Close', 'ROC']
    )
    new_dataset['ROC'][0] = 0

    for i in range(1, len(data)):
        new_dataset['Date'][i] = data['Date'][i]
        new_dataset['Close'][i] = data['Close'][i]
        new_dataset['ROC'][i] = (
            (data['Close'][i] / data['Close'][i - 1]) - 1
        ) * 100

    new_dataset.index = new_dataset.Date
    new_dataset.drop('Date', axis=1, inplace=True)

    return new_dataset


def calculate_roc(df: DataFrame, feature='Close') -> DataFrame:
    new_df = df.copy()
    new_df['ROC'] = 0.0

    for i in range(1, len(new_df)):
        new_df['ROC'][
            i] = (new_df[feature][i] / new_df[feature][i - 1]) * 100 - 100

    print(new_df['ROC'])

    return new_df
