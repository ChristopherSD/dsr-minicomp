import pandas as pd
import numpy as np

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
    df = pd.read_csv('./data/clean_data.csv')

    #Drop values - if still any exists
    #df = df.dropna(axis=1)

    #Save output
    df.to_csv('./data/model_input_data.csv', index=False)

    return df.dropna(axis=1)
