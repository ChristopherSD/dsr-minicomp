import pandas as pd
import pickle

from data_transformation import get_all_train_data, create_basetable
from feature_engineering import *


def get_input_data():
    # update data
    _ = create_basetable(get_all_train_data())
    _ = execute_feature_engineering_all()

    # load data
    df = pd.read_csv('./data/model_input_data.csv')

    return df


def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])


def score_model_ti():

    x_test, y_test = prepare_data_for_model_ti()

    pkl_filename = "./models/model_tom_rf_rmpse.pkl"
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    y_pred = pickle_model.predict(x_test)

    print(print(metric(y_pred, np.array(y_test))))

    return x_test, y_test, y_pred