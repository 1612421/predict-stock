from h5py._hl import dataset
import pandas as pd


def parseCloseDF(data_file_path):
    df = pd.read_csv(data_file_path)
    df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df.index = df['Date']
    data = df.sort_index(ascending=True, axis=0)
    new_dataset = pd.DataFrame(index=range(0, len(df)),
                               columns=['Date', 'Close'])

    for i in range(0, len(data)):
        new_dataset['Date'][i] = data['Date'][i]
        new_dataset['Close'][i] = data['Close'][i]

    new_dataset.index = new_dataset.Date
    new_dataset.drop('Date', axis=1, inplace=True)

    return new_dataset
