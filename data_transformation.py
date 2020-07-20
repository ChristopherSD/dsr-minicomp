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


def fillna_StoreType_and_factorize(all_data):
    '''Fill nan value in StoreType with string 'unknown' and label encode the values
    
    Args:
        all_data: Raw data
    Returns:
        output: New column
        int_to_storetype: Decoding dictionary
        storetype_to_int: Encoding dictionary
    '''
    
    output = all_data.StoreType.fillna('unknown')
    storetype_to_int = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'unknown': 0}
    int_to_storetype = dict([(v,k) for k,v in storetype_to_storecode.items()])

    output = output.map(storetype_to_int)
    
    return output, int_to_storetype, storetype_to_int

