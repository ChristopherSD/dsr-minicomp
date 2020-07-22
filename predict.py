#!/usr/bin/env
import os
import predictChris
from feature_engineering import *
from joblib import dump, load
from testing.predictiveAccuracy import metric


def main():
    test_err = predictChris.predict()


def predict_michael():

    model1_path = os.path.join('./models', 'rf_2')
    y_pred1 = np.load(os.path.join(model1_path, 'y_pred_TEST.npy'))

    # get test data
    test_data = get_all_test_data()

    # drop rows with Sales == 0
    test_data = test_data.loc[test_data.Sales > 0, :]
    test_data.sort_values(by='Date', inplace=True)

    oneh_enc = load(os.path.join(model1_path, 'm_onehotencoder.joblib'))
    target_enc = load(os.path.join(model1_path, 'm_targetencoder.joblib'))
    test_data, _, _ = my_preprocess_data(test_data, oneh_enc, target_enc)

    y_test = test_data[['Sales']].values

    score = metric(y_test, y_pred1)

    print(f'Test score: {score}')


if __name__ == "__main__":
    main()
