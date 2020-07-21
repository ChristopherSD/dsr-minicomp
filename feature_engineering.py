import pandas as pd
import datetime as datetime
import numpy as np


def execute_feature_engineering() -> pd.DataFrame:
    """
    Main function to execute all feature_e

    Returns:

    """
    #Load clean data
    df = pd.read_csv('./data/clean_data.csv')
    df.Date = pd.to_datetime(df.Date)

    # CompetitionSince
    generate_CompetitionSince(df)

    # Promo2SinceNWeeks
    generate_Promo2SinceNWeeks(df)

    assert df.isna().sum() == 0

    #Save output
    df.to_csv('./data/model_input_data.csv', index=False)

    return df


def generate_CompetitionSince(all_data: pd.DataFrame):
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
    all_data.drop(labels=['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], axis=1, inplace=True)


def generate_Promo2SinceNWeeks(all_data: pd.DataFrame):
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
    all_data.drop(labels=['Promo2SinceYear', 'Promo2SinceWeek'], axis=1, inplace=True)

