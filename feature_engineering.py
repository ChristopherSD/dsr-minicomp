import pandas as pd
import datetime as datetime
import numpy as np
from data_transformation import create_basetable

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

def execute_feature_engineering() -> pd.DataFrame:

    #Load clean data
    _ = create_basetable()
    df = pd.read_csv('./data/train.csv')

    #Drop values - if still any exists
    df = df.dropna(axis=1)

    #Save output
    df.to_csv('./data/model_input_data.csv')

    return df.dropna(axis=1)
