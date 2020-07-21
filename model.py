import pandas as pd

from data_transformation import create_basetable
from feature_engineering import execute_feature_engineering


def get_input_data():
    # update data
    _ = create_basetable()
    _ = execute_feature_engineering()

    # load data
    df = pd.read_csv('./data/model_input_data.csv')

    return df