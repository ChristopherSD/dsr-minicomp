"""Functions to impute and transform data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
from tqdm import tqdm


_sales_col = 'Sales'


def get_all_train_data():
    """Get all data an merge into one big dataframe and convert Date to datetime
    """
    train_path = Path(Path(__file__).parent.absolute(), 'data', 'train.csv')
    store_path = Path(Path(__file__).parent.absolute(), 'data', 'store.csv')
    train_data = pd.read_csv(train_path)
    store_data = pd.read_csv(store_path)

    all_data = train_data.merge(store_data, how='outer', left_on='Store', right_on='Store')
    all_data.Date = pd.to_datetime(all_data.Date, format='%Y-%m-%d')

    return all_data

def get_all_test_data():
    """Get all TEST data an merge into one big dataframe and convert Date to datetime
    """
    test_path = Path(Path(__file__).parent.absolute(), 'data', 'test.csv')
    store_path = Path(Path(__file__).parent.absolute(), 'data', 'store.csv')
    test_data = pd.read_csv(test_path)
    store_data = pd.read_csv(store_path)

    all_data = test_data.merge(store_data, how='outer', left_on='Store', right_on='Store')
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


def drop_empty_and_zero_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all rows where the Sales column is Na or zero.
    """
    return df[(df[_sales_col].notna()) & (df[_sales_col] > 0)]


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
    dow_imputed = df[dow_col].fillna(missing_dow[dow_col] + 1.0)

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


def create_basetable(df, impute_config={
    'Store': 0,
    'StoreType': 'unknown',
    'SchoolHoliday': 'unknown',
    'Assortment': 'unknown',
    'StateHoliday': 'unknown',
    'DayOfWeek': 'unknown',
    'Promo': 'unknown',
    'Promo2': 'unknown',
    'CompetitionDistance': -1
}
                     ) -> pd.DataFrame:
    """Prepare base table:
        - Drop sales
        - Make imputation with default value
        - Make customized imputations

        Args:
            df: The data frame that is to be cleaned
        Returns:
            df: Cleanse data ready for FeatureEngineering
    """

    # competition modification - dropping NULL sales
    df = drop_empty_and_zero_sales(df)

    # custom imputers

    # DayOfWeek
    df.DayOfWeek = impute_dayofweek_from_date(df)

    # Customers
    inplace_impute_rolling_avg_customers(df)

    # Open (is open)
    df.Open = impute_open_from_customers(df)

    # StateHoliday
    df.StateHoliday = impute_holiday(df, 'StateHoliday')

    # impute default values
    for col, default_value in zip(impute_config.keys(), impute_config.values()):
        df[col] = df[col].fillna(default_value)

    # save output
    clean_path = Path(Path(__file__).parent.absolute(), 'data', 'clean_data.csv')
    df.to_csv(clean_path, index=False)

    return df


def impute_holiday(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Impute holiday indicator based on information from another stores at that specific day.

    Args:
        df (pd.DataFrame): input DataFrame
        column: column that requires imputed values
    Returns:
        pd.Series: imputed column with holiday indicator
    """
    df = df[[column, 'Date']]
    if column == 'StateHoliday':
        df[column] = df[column].replace(0.0, 0)

    df[column] = df[column].astype(str)

    df_aggr = df.groupby('Date').agg(lambda x: x.value_counts().index[0])
    null_mask = (df[column] == 'nan')

    df.loc[null_mask, 'StateHoliday'] = df[null_mask]['Date'].apply(lambda x: df_aggr.to_dict()['StateHoliday'][x]).values

    return df[column]


def inplace_impute_rolling_avg_customers(all_data: pd.DataFrame, do_plot=False):
    """Manipulates input Dataframe in place:
    Fills in missing Customers with a rolling average from the last monthly,
    for each weekday respectively. This is done for each store ID separately.
    Where not possible, a simple median is used as fill value.
    """

    customers_per_store_and_date = all_data.groupby(['Store', 'Date'])['Customers'].mean()

    # go through store
    store_numbers = customers_per_store_and_date.index.get_level_values('Store').unique().astype(int)
    pbar = tqdm(store_numbers)
    for store_i in store_numbers:
        store_data = customers_per_store_and_date.loc[(store_i, slice(None))]

        # slice at store index and resampl per day (missing days will be nan)
        tmp = store_data.droplevel(0).resample('d').mean().to_frame()

        # average over the last month for each weekday, respectively
        fill_vals = np.nanmean(pd.concat([tmp.shift(7), tmp.shift(14), tmp.shift(21), tmp.shift(28)], axis=1), axis=1)

        # fill in rolling average, if not possible just fill in average over the whole training set
        tmp['rolling_avg'] = fill_vals
        tmp.loc[(tmp.rolling_avg.isna()) & (tmp.index.weekday != 6), 'rolling_avg'] = tmp[
            tmp.index.weekday != 6].Customers.median()
        tmp.loc[(tmp.rolling_avg.isna()) & (tmp.index.weekday == 6), 'rolling_avg'] = tmp[
            tmp.index.weekday == 6].Customers.median()

        # substitute average into missing rows in base table
        na_rows = store_data.loc[store_data.isna()].index.get_level_values('Date')
        all_data.loc[(all_data.Store == store_i) & (all_data.Customers.isna()), 'Customers'] = tmp.loc[
            na_rows, 'rolling_avg'].values

        pbar.update(1)
    pbar.close()

    # fill in rest with mean of Sunday/Weekday respectively
    all_data.loc[(all_data.Customers.isna()) & (all_data.Date.dt.weekday == 6), 'Customers'] = all_data.loc[
        all_data.Date.dt.weekday == 6].Customers.median()
    all_data.loc[(all_data.Customers.isna()) & (all_data.Date.dt.weekday != 6), 'Customers'] = all_data.loc[
        all_data.Date.dt.weekday != 6].Customers.median()

    # optional plot
    if do_plot:
        fig, ax = plt.subplots(figsize=(15, 5))

        after = all_data.groupby(['Store', 'Date'])['Customers'].mean().loc[(store_i, slice(None))]
        after = after.reset_index()

        after.plot(ax=ax, x='Date', y='Customers', marker='o', linewidth=0)
        tmp['Customers'].plot(ax=ax, marker='o')

        ax.set_xlim([datetime.date(2013, 2, 1), datetime.date(2013, 3, 1)])

        for i in range(60):
            plt.axvline(datetime.date(2013, 2, 1) + datetime.timedelta(days=i), alpha=0.2)
