import pandas as pd
import datetime as datetime
import numpy as np
from dateutil.parser import parse


def generate_CompetitionSince(all_data: pd.DataFrame):
    """Generate (inplace) a feature 'CompetitionSince' which counts the months (in integer) since
    when the competition started.
    """

    mask = ~all_data.CompetitionOpenSinceYear.isna()
    year = all_data.loc[mask, 'CompetitionOpenSinceYear'].astype(np.int).astype(str)
    month = all_data.loc[mask, 'CompetitionOpenSinceMonth'].astype(np.int).apply('{:02d}'.format)
    now_date = all_data.loc[mask, 'Date']

    CompetitionSince = (now_date.dt.to_period('M') - pd.to_datetime(year + '-' + month, format='%Y-%m').dt.to_period('M'))
    CompetitionSince = CompetitionSince.apply(lambda x: x.n)

    all_data.loc[mask, 'CompetitionSince'] = CompetitionSince


def execute_transformations(df: pd.DataFrame) -> pd.DataFrame:

    return df.dropna(axis=1)


def is_in_promo_month(row, itvl_col='PromoInterval'):
    if (itvl_col in row) and isinstance(row[itvl_col], str):
        intervals = row[itvl_col].split(',')
        itvl_dates = list(map(parse, intervals))
        for date in itvl_dates:
            if row['Date'].month == date.month:
                return 1.0

    return 0.0


def generate_PromoStarted(all_data: pd.DataFrame, itvl_col='PromoInterval'):
    """Generate (inplace) a feature 'CompetitionSince' which counts the months (in integer) since
    when the competition started.
    """
    new_col_name = 'PromoStarted'
    promo_started = all_data.apply(is_in_promo_month, axis=1)
    all_data[new_col_name] = promo_started

    all_data.drop(labels=[itvl_col], axis=1, inplace=True)
