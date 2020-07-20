"""Functions to impute and transform data.
"""
from typing import List

import pandas as pd

_sales_col = 'Sales'

def get_all_train_data():
    """Get all data an merge into one big dataframe and convert Date to datetime
    """
    train_data = pd.read_csv("data/train.csv")
    store_data = pd.read_csv("data/store.csv")

    all_data = train_data.merge(store_data, how='outer', left_on='Store', right_on='Store')
    all_data.Date = pd.to_datetime(all_data.Date, format='%Y-%m-%d')

    return all_data


def fillna_StoreType_and_factorize(all_data):
    """Fill nan value in StoreType with string 'unknown' and label encode the values

    Args:
        all_data: Raw data
    Returns:
        output: New column
        int_to_storetype: Decoding dictionary
        storetype_to_int: Encoding dictionary
    """

    output = all_data.StoreType.fillna('unknown')
    storetype_to_int = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'unknown': 0}
    int_to_storetype = dict([(v, k) for k, v in storetype_to_int.items()])

    output = output.map(storetype_to_int)

    return output, int_to_storetype, storetype_to_int


def drop_empty_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all rows where the Sales column is Na.
    """
    return df[df[_sales_col].notna()]


def impute_dayofweek_from_date(df: pd.DataFrame, date_col='Date', dow_col='DayOfWeek') -> pd.Series:
    """ Impute the day of the week from the corresponding date, if present.
    If the date column is not of type DateTime, it will be converted locally (the original data frame is not changed).
    If the corresponding date entry is not available, day of week will remain Na.

    Args:
        df: original raw data
        date_col: The name of the column containing the date (as String or DateTime)
        dow_col: The name of the column containing the day of week (as float)
    Returns:
        dow_imputed: A pd.Series representing the day of week with imputed values.
    """
    missing_dow = df.loc[df[dow_col].isna()]
    if not pd.api.types.is_datetime64_any_dtype(missing_dow[date_col]):
        missing_dow[date_col] = pd.to_datetime(missing_dow[date_col])

    missing_dow[dow_col] = missing_dow[date_col].dt.dayofweek
    # add one to dt.dayofweek since the original original feature values start with Monday == 1,
    # whereas DateTime starts with Monday == 0
    dow_imputed = df[dow_col].fillna(missing_dow[dow_col]) + 1.0

    return dow_imputed


def create_basetable(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare base table:
        - Drop sales
        - Make base imputation
    """

    # general
    df = drop_empty_sales(df)

    ########################################
    # Data Imputation and transformation
    ########################################
    df = fillna_StoreType_and_factorize(df)

    return df
