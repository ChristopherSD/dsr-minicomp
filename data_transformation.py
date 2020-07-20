'''Function to impute and transform data.
'''

import pandas as pd

def get_all_train_data():
    '''Get all data an merge into one big dataframe
    '''
    train_data = pd.read_csv("data/train.csv")
    store_data = pd.read_csv("data/store.csv")

    all_data = train_data.merge(store_data, how = 'outer', left_on='Store', right_on='Store')
    return all_data