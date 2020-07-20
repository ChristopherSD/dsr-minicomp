'''Function to impute and transform data.
'''
from typing import List

import pandas as pd

_sales_col = 'Sales'

def get_all_train_data():
    '''    Get all data an merge into one big dataframe
    '''
    train_data = pd.read_csv("data/train.csv")
    store_data = pd.read_csv("data/store.csv")

    all_data = train_data.merge(store_data, how = 'outer', left_on='Store', right_on='Store')
    all_data.Date = pd.to_datetime(all_data.Date, format='%Y-%m-%d')

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
    int_to_storetype = dict([(v, k) for k, v in storetype_to_int.items()])

    output = output.map(storetype_to_int)

    return output, int_to_storetype, storetype_to_int


def drop_empty_sales(df: pd.DataFrame) -> pd.DataFrame:
    '''Drop all rows where the Sales column is Na.
    '''
    return df[df[_sales_col].notna()]

def create_basetable(df: pd.DataFrame) -> pd.DataFrame:
    
    """
    Prepare base table:
        - Drop sales
        - Make base imputation
    """
    
    #general 
    df = drop_empty_sales(df)
    
    ########################################
    #Data Imputation and transformation
    ########################################
    df = fillna_StoreType_and_factorize(df)
    
    return df