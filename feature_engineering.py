import pandas as pd
import datetime as datetime
import numpy as np
from dateutil.parser import parse
from pathlib import Path
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import OneHotEncoder
from data_transformation import *
import tqdm


def generate_CompetitionSince(all_data: pd.DataFrame, drop=True):
    """Generate (inplace) a feature 'CompetitionSince' which counts the months (in integer) since
    when the competition started.
    Fills missing values with -1000.
    Creates a new boolean column 'Competition_missing' highlighting the missing values.
    """

    mask = ~all_data.CompetitionOpenSinceYear.isna()
    year = all_data.loc[mask, 'CompetitionOpenSinceYear'].astype(np.int).astype(str)
    month = all_data.loc[mask, 'CompetitionOpenSinceMonth'].astype(np.int).apply('{:02d}'.format)
    now_date = all_data.loc[mask, 'Date']

    CompetitionSince = (now_date.dt.to_period('M') -
                        pd.to_datetime(year + '-' + month, format='%Y-%m').dt.to_period('M'))
    CompetitionSince = CompetitionSince.apply(lambda x: x.n)

    all_data.loc[mask, 'CompetitionSince'] = CompetitionSince

    all_data.loc[:, 'Competition_missing'] = all_data.CompetitionSince.isna()
    all_data.CompetitionSince.fillna(-1000, inplace=True)

    if drop:
        all_data.drop(labels=['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], axis=1, inplace=True)


def execute_feature_engineering_all(df: pd.DataFrame) -> pd.DataFrame:

    # CompetitionSince
    generate_CompetitionSince(df)

    # Promo2SinceNWeeks
    generate_Promo2SinceNWeeks(df)

    # PromoStarted
    generate_PromoStarted(df)

    # Drop values - if still any exists
    # df = df.dropna(axis=1)

    # Save output
    input_data_path = Path(Path(__file__).parent.absolute(), 'data', 'model_input_data.csv')
    df.to_csv(input_data_path, index=False)

    return df.dropna(axis=1)


def one_hot_encoder_fit_transform(df: pd.DataFrame, col_name: str):
    """
    Function to fit and transform column in DataFrame with OneHotEncoder

    Args:
        df - DataFrame to transform
        col_name: name of the column that has to be transformed
    Returns:
        input DataFrame with concatenated, transformed column
    """
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

    enc.fit(df[col_name].values.reshape(-1, 1))

    return one_hot_encoder_transform(df, col_name, enc), enc


def one_hot_encoder_transform(df: pd.DataFrame, col_name: str, enc):
    """
    Function to fit and transform column in DataFrame with OneHotEncoder

    Args:
        df: DataFrame to transform
        col_name: name of the column that has to be transformed
        enc: instance of fitted OneHotEncoder
    Returns:
        input DataFrame with concatenated, transformed column
    """
    encoded_column = pd.DataFrame(enc.transform(df[col_name].values.reshape(-1, 1)),
                                  columns=[col_name + '_' + str(item) for item in range(len(enc.categories_[0]))],
                                  index=df.index

                                  )
    return pd.concat([df, encoded_column], axis=1).drop(col_name, axis=1)


def is_StateHoliday(df):
    """Generates a new boolean column, if it is a StateHoliday or not
    """
    return ((df.StateHoliday == 'a') | (df.StateHoliday == 'b') | (df.StateHoliday == 'c'))

def log_transform(inp: pd.Series):
    """
    Function to log transform - takes care of negative and 0 values.

    Args:
        inp - pd.Series to log transform
    Returns:
        transformed pd.Series
    """
    x = pd.Series()
    x = inp - inp.min() + 1
    return np.log(x)


def is_in_promo_month(row, itvl_col='PromoInterval'):
    if (itvl_col in row) and isinstance(row[itvl_col], str):
        intervals = row[itvl_col].split(',')
        itvl_dates = list(map(parse, intervals))
        for date in itvl_dates:
            if row['Date'].month == date.month:
                return 1.0

    return 0.0


def generate_PromoStarted(all_data: pd.DataFrame, drop=True, itvl_col='PromoInterval'):
    """Generate (inplace) a feature 'CompetitionSince' which counts the months (in integer) since
    when the competition started.
    """
    new_col_name = 'PromoStarted'
    promo_started = all_data.apply(is_in_promo_month, axis=1)
    all_data[new_col_name] = promo_started

    if drop:
        all_data.drop(labels=[itvl_col], axis=1, inplace=True)


def generate_Promo2SinceNWeeks(all_data: pd.DataFrame, drop=True):
    """Generate (inplace) a feature 'Promo2SinceNWeeks' which counts the weeks (in integer) since
    when a Promo2 started.
    Fills missing values with -1000.
    Creates a new boolean column 'Promo2SinceNWeeks_missing' highlighting the missing values.
    """

    mask = ~all_data.Promo2SinceYear.isna()
    year = all_data.loc[mask, 'Promo2SinceYear'].astype(np.int).astype(str)
    week = all_data.loc[mask, 'Promo2SinceWeek'].astype(np.int).apply('{:02d}'.format)
    now_date = all_data.loc[mask, 'Date']

    Promo2SinceNWeeks = (now_date.dt.to_period('W') -
                         pd.to_datetime(year + '-' + week + '0', format='%Y-%W%w').dt.to_period('W'))
    Promo2SinceNWeeks = Promo2SinceNWeeks.apply(lambda x: x.n)

    all_data.loc[mask, 'Promo2SinceNWeeks'] = Promo2SinceNWeeks

    all_data.loc[:, 'Promo2SinceNWeeks_missing'] = all_data.Promo2SinceYear.isna()
    all_data.Promo2SinceNWeeks.fillna(-1000, inplace=True)

    if drop:
        all_data.drop(labels=['Promo2SinceYear', 'Promo2SinceWeek'], axis=1, inplace=True)


def generate_col_month(df):
    """Generates a new feature "month"
    """
    month = df.Date.dt.month
    return month


def target_encode_Stores(df, enc=None):
    """Target encode the Store variable using the category_encoders module

    Args:
        df: Data
        enc: Existing Encoder / if None retrain new encoder
    """

    target = df['Sales'].values
    stores = df['Store'].astype(str)

    if not enc:
        print("Fit TargetEncoder...")
        enc = TargetEncoder()
        new_store = enc.fit_transform(stores, target)
    else:
        print("Transform using existing TargetEncoder...")
        new_store = enc.transform(stores, target)

    df.loc[:, 'Store'] = new_store

    return new_store, enc


def target_encode_custom(df: pd.DataFrame, name: str, enc=None):
    """Target encode the Store variable using the category_encoders module

    Args:
        df: Data
        name (str): name of the column to encode
        enc: Existing Encoder / if None retrain new encoder
    """

    target = df['Sales'].values
    stores = df[name].astype(str)

    if not enc:
        print("Fit TargetEncoder...")
        enc = TargetEncoder()
        new_store = enc.fit_transform(stores, target)
    else:
        print("Transform using existing TargetEncoder...")
        new_store = enc.transform(stores, target)

    df.loc[:, name] = new_store

    return new_store, enc

def generate_cyclic_feature_month(df):
    """Generates a new feature "month"
    """
    sin_month = np.sin(df.Date.dt.month/12*2*np.pi)
    cos_month = np.cos(df.Date.dt.month/12*2*np.pi)
    sin_month = sin_month.reindex(df.index)
    cos_month = cos_month.reindex(df.index)
    return sin_month, cos_month

def generate_cyclic_feature_week(df):
    """Generates a new feature "Week"
    """
    sin_week = np.sin(df.Date.dt.week/52*2*np.pi)
    cos_week = np.cos(df.Date.dt.week/52*2*np.pi)
    sin_week = sin_week.reindex(df.index)
    cos_week = cos_week.reindex(df.index)
    return sin_week, cos_week

def my_impute_data(data):
    """Custom function for Michael's Model
    """

    df = data.copy()
    # all_data = create_basetable()
    # all_data = pd.read_csv('base_table.csv')

    # Promo2
    df.Promo2.fillna('unknown', inplace=True)

    # CompetitionDistance
    impute_competition_distance = df.CompetitionDistance.median()
    df.CompetitionDistance.fillna(impute_competition_distance, inplace=True)

    # CompetitionSince
    generate_CompetitionSince(df)

    # Promo2SinceNweeks
    generate_Promo2SinceNWeeks(df)

    # PromoInterval
    df.drop(labels=['PromoInterval'], axis=1, inplace=True)

    # StateHoliday
    new_col = df[['StateHoliday']].apply(lambda x: x['StateHoliday'] if x['StateHoliday'] in ['a', 'b', 'c'] else '0',
                                         axis=1)
    df.StateHoliday = new_col

    return df


def massive_onehot(input_df, enc=None):
    """Apply OnehotEncoder to columns:
    'Promo', 'SchoolHoliday', 'StateHoliday', 'StoreType', 'Assortment', 'Promo2'
    """

    data = input_df.copy()

    cat_features = ['Promo', 'SchoolHoliday', 'StateHoliday', 'StoreType', 'Assortment', 'Promo2']

    # replace floats (0.0, 1.0) with strings ('0', '1')
    for c in cat_features:
        data[c] = data[c].replace(0, '0').replace(1, '1')

    if not enc:
        print("Fit OneHotEncoder...")
        enc = OneHotEncoder()
        new_cat = pd.DataFrame(enc.fit_transform(data[cat_features].astype(str)).toarray())
    else:
        print("Transform using existing OneHotEncoder...")
        enc = enc
        new_cat = pd.DataFrame(enc.transform(data[cat_features].astype(str)).toarray())

    new_cat.columns = enc.get_feature_names()

    new_cat.index = data.index

    data = data.join(new_cat).drop(cat_features, axis=1)

    return data, enc


def my_preprocess_data(data, oneh_enc=None, target_enc=None):
    """ Data preprocessing function for Michael's Model
    """

    data = create_basetable(data)

    data = my_impute_data(data)

    # data = engineer_simple_features(data)
    data, oneh_enc = massive_onehot(data, oneh_enc)

    new_store, target_enc = target_encode_Stores(data, target_enc)

    sin_month, cos_month = generate_cyclic_feature_month(data)
    data['sin_month'] = sin_month
    data['cos_month'] = cos_month
    data = data.drop(['Date'], axis=1)

    return data, oneh_enc, target_enc
