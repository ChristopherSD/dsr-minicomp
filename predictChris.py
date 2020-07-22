import pandas as pd
import xgboost as xgb
import feature_engineering as fe
import matplotlib.pyplot as plt
from pathlib import Path
from featurePipelineChris import transform_test_data
from testing.predictiveAccuracy import metric

def predict():
    data_path = Path(Path(__file__).parent.absolute(), 'data', 'test.csv')
    store_path = Path(Path(__file__).parent.absolute(), 'data', 'store.csv')
    model_path = Path(Path(__file__).parent.absolute(), 'models', 'xgboost_chris.model')

    test_data = pd.read_csv(data_path)
    store_data = pd.read_csv(store_path)
    all_data = test_data.merge(store_data, how='outer', left_on='Store', right_on='Store')
    all_data.Date = pd.to_datetime(all_data.Date, format='%Y-%m-%d')
    train = transform_test_data(all_data)

    y = train['Sales']
    X = train.drop(['Date', 'Sales'], axis=1)

    xg_reg = xgb.XGBRegressor()
    xg_reg.load_model(model_path)

    preds = xg_reg.predict(X)

    rmspe = metric(preds, y.to_numpy())
    print(f"RMSPE on test data set with XGBoost: {rmspe} /%")
    '''
    xgb.plot_tree(xg_reg,num_trees=0)
    plt.rcParams['figure.figsize'] = [50, 10]
    plt.show()

    xgb.plot_importance(xg_reg)
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.show()
    '''
    return rmspe
