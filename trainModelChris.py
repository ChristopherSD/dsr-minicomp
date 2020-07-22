import pandas as pd
from xgboost import XGBClassifier

import data_transformation as dt
import feature_engineering as fe
import pickle
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from pathlib import Path
from testing.predictiveAccuracy import metric
from testing.predictiveAccuracy import custom_scorer_metric
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn import linear_model

load = False

##############
#  xgboost   #
##############

clean_path = Path(Path(__file__).parent.absolute(), 'data', 'clean_data.csv')

if load:
    df = pd.read_csv(clean_path)
    df.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')
    input_data_path = Path(Path(__file__).parent.absolute(), 'data', 'model_input_data.csv')
    data = pd.read_csv(input_data_path)
    data.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')
else:
    raw_data = dt.get_all_train_data()
    df = dt.create_basetable(raw_data) # saves data
    data = fe.execute_feature_engineering_all(df)

# Data Cleaning and Feature Engineering
data = data.drop(['Promo2SinceNWeeks_missing', 'Competition_missing'], axis=1)
data['Promo2'] = data['Promo2'].astype('bool')
data['Promo'] = data['Promo'].astype('bool')

sin_month, cos_month = fe.generate_cyclic_feature_month(data)
sin_week, cos_week = fe.generate_cyclic_feature_week(data)
data['sin_month'] = sin_month
data['cos_month'] = cos_month
data['sin_week'] = sin_week
data['cos_week'] = cos_week

data['is_state_holiday'] = fe.is_StateHoliday(data)
data['is_school_holiday'] = fe.is_SchoolHoliday(data)
data.drop(['StateHoliday', 'SchoolHoliday'], axis=1, inplace=True)

# label encoding
categorical_columns = ['Promo', 'StoreType', 'Assortment', 'Promo2']
le = LabelEncoder()
for col in categorical_columns:
    data[col] = le.fit_transform(data[col])

# train XGBoost
y = data['Sales']
X = data.drop(['Date', 'Sales'], axis=1)

data_dmatrix = xgb.DMatrix(data=X, label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


params = {
    'colsample_bytree': 0.3,
    'learning_rate': 0.07,
    'max_depth': 5,
    'alpha': 10,
    'min_child_weight': 2,
    'n_estimators': 100,
    'reg_alpha': 0.75,
    'reg_lambda': 0.45,
    'subsample': 0.5
}


cv_results = xgb.cv(
    dtrain=data_dmatrix,
    params=params,
    nfold=3,
    num_boost_round=50,
    early_stopping_rounds=10,
    metrics="rmse",
    as_pandas=True
)

print(cv_results.head())
print((cv_results["test-rmse-mean"]).tail(1))

xg_reg = xgb.XGBRegressor(params=params)
xg_reg.fit(X_train, y_train)

model_path = Path(Path(__file__).parent.absolute(), 'models', 'xgboost_chris.model')
xg_reg.save_model(model_path)

# Prediction
preds = xg_reg.predict(X_test)

rmspe = metric(preds, y_test.to_numpy())
print(rmspe)

xgb.plot_tree(xg_reg,num_trees=0)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()

xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()

