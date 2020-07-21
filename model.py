import pandas as pd

from data_transformation import get_all_train_data, create_basetable
from feature_engineering import execute_feature_engineering_all


def get_input_data():
    # update data
    _ = create_basetable(get_all_train_data())
    _ = execute_feature_engineering_all()

    # load data
    df = pd.read_csv('./data/model_input_data.csv')

    return df