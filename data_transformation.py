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
    Does not take Date Na values into account.

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


def impute_open_from_customers(df: pd.DataFrame, open_col='Open', customers_col='Customers') -> pd.Series:
    """ Impute missing open values based on customer number.
    If customer number in the same entry is greater than zero, the shop must have been open.
    Does not take Customers Na values into account.

    Args:
        df: original raw data
        open_col: The name of the column containing whether the store was opened on a given date (0 or 1)
        customers_col: The name of the column containing the numbers of customers on a given date
    Return:
         open_imputed: A pd.Series representing the open status of a shop with imputed values.
    """
    missing_open = df.loc[df[open_col].isna() & df[customers_col] > 0]
    missing_open[open_col] = 1.0
    open_imputed = df[open_col].fillna(missing_open[open_col])

    missing_closed = df.loc[df[open_col].isna() & df[customers_col] <= 0]
    missing_closed[open_col] = 0.0
    open_imputed = open_imputed.fillna(missing_closed[open_col])

    return open_imputed


def create_basetable() -> pd.DataFrame:
    """Prepare base table:
        - Drop sales
        - Make imputation with default value
        - Make customized imputations

        Args:
        Returns:
            df: Cleanse data ready for FeatureEngineering
    """
    #get raw data
    df = get_all_train_data()

    #competition modification - dropping NULL sales
    df = drop_empty_sales(df)

    #custom imputers
    df['WeekOfDay'] = impute_dayofweek_from_date

    #default values
    impute_config = {
        'Store': 0,
        'DayOfWeek': 'unknown',
        'Promo': ''
    }
    for col, default_value in zip(impute_config.keys(), impute_config.values()):
        df[col] = df[col].fillna(default_value)

    return df
