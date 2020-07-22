import pandas as pd
import data_transformation as dt
import feature_engineering as fe
import pickle
import matplotlib.pyplot as plt
from testing.predictiveAccuracy import custom_scorer_metric
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn import linear_model
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from testing.predictiveAccuracy import metric

'''
train_data = pd.read_csv("../data/train.csv")
store_data = pd.read_csv("../data/store.csv")

all_data = train_data.merge(store_data, how = 'outer', left_on='Store', right_on='Store')
all_data.Date = pd.to_datetime(all_data.Date, format='%Y-%m-%d')
'''
'''
# test generate_PromoStarted
#data = all_data.copy()
#generate_PromoStarted(data[50:55])
#print(data.columns)


# test impute_dayofweek_from_date
md = impute_dayofweek_from_date(all_data)
data = all_data.copy()
data['DayOfWeek'] = md
print(data['DayOfWeek'].unique())

op = impute_open_from_customers(data)
print(data.isna().sum())
'''


load = False

#raw_data = dt.get_all_train_data()
#raw_data = all_data


'''
sorted_data = raw_data.sort_values(by=['Date'])

target_col = 'Sales'

# make time series split for cross validation

kf = TimeSeriesSplit(n_splits=5)
X_train_folds = []
X_test_folds = []
y_train_folds = []
y_test_folds = []
for train_index, test_index in kf.split(sorted_data):
    train_data = raw_data.iloc[train_index, :]
    y_train_folds.append(train_data)
    X_train_folds.append(train_data)

    test_data = raw_data.iloc[test_index, :]
    y_test_folds.append(test_data)
    X_test_folds.append(test_data)

# apply imputing to each set
if not load:
    X_train_folds_imp = []
    X_test_folds_imp = []
    y_train_folds_imp = []
    y_test_folds_imp = []

    for fold in X_train_folds:
        X_train_folds_imp.append(dt.create_basetable(fold))
    for fold in X_test_folds:
        X_test_folds_imp.append(dt.create_basetable(fold))
    for fold in y_train_folds:
        y_train_folds_imp.append(dt.create_basetable(fold))
    for fold in y_test_folds:
        y_test_folds_imp.append(dt.create_basetable(fold))

split_clean_data_path = '../data/chris_split_clean_data_list.pkl'

if not load:
    with open(split_clean_data_path, 'wb') as f:
        pickle.dump([X_train_folds_imp, X_test_folds_imp, y_train_folds_imp, y_test_folds_imp], f)

if load:
    saved_folds = pickle.load(open(split_clean_data_path, "rb"))
    X_train_folds_imp = saved_folds[0]
    X_test_folds_imp = saved_folds[1]
    y_train_folds_imp = saved_folds[2]
    y_test_folds_imp = saved_folds[3]

# grid search on linear regression
model = linear_model.LinearRegression()
params = {
    'fit_intercept': [True, False],
    'normalize': [True, False]
}
grid = GridSearchCV(estimator=model, param_grid=params,
                    scoring=make_scorer(custom_scorer_metric, greater_is_better=False),
                    cv=kf.split(sorted_data), verbose=True)

# fold sizes
for folds in [X_train_folds_imp, X_test_folds_imp, y_train_folds_imp, y_test_folds_imp]:
    for fold in folds:
        print(fold.shape)

print(len(X_train_folds_imp[-1]) + len(X_test_folds_imp[-1]))

# restack imputed folds for the gridsearch
restacked_X_data = X_train_folds_imp[-1].append(X_test_folds_imp[-1])
restacked_y_data = y_train_folds_imp[-1].append(y_test_folds_imp[-1])

# pd.get_dummies(restacked_X_data).dtypes

grid.fit(
    restacked_X_data.drop(['Date', 'Sales'], axis=1).to_numpy(),
    restacked_y_data['Sales'].to_numpy()
)
'''


'''
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X_train_no_date, y_train)

preds = regr.predict(X_test_no_date)

# random forest prediction
metric(preds, y_test.to_numpy())

# average prediction
metric(np.mean(X_test_no_date.to_numpy(), axis=1), y_test.to_numpy())
'''

##############
#  xgboost   #
##############

if not load:
    raw_data = dt.get_all_train_data()
    data = dt.create_basetable(raw_data)
    data.to_csv('./data/clean_data.csv')

data = fe.execute_feature_engineering_all()

# label encoding
categorical_columns = []
le = LabelEncoder()



y = raw_data['Sales']
X = raw_data.drop(['Sales'], axis=1)

data_dmatrix = xgb.DMatrix(data=X, label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


xg_reg = xgb.XGBRegressor(
    objective='reg:linear',
    colsample_bytree=0.3,
    learning_rate=0.1,
    max_depth=5,
    reg_alpha=10,
    n_estimators=10
)

xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)

rmspe = metric(preds, y_test)

xgb.plot_tree(xg_reg,num_trees=0)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()

xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()