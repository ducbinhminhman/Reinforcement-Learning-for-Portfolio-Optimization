import pandas as pd

def load_data(file_path):
    """
    Load portfolio data, normalize, and split into training, validation, and test sets.
    """
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, format='%Y%m')
    df = df / 100

    train_data = df[df.index.year <= 2010]
    valid_data = df[(df.index.year > 2010) & (df.index.year <= 2015)]
    test_data = df[df.index.year > 2015]
    
    return train_data.values, valid_data.values, test_data.values, train_data.index, valid_data.index, test_data.index
