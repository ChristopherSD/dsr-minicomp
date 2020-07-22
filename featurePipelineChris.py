import pandas as pd
import feature_engineering as fe
import data_transformation as dt
from sklearn.preprocessing import LabelEncoder
from pathlib import Path


def transform_test_data(df: pd.DataFrame) -> pd.DataFrame:
    load = False

    if load:
        data_path = Path(Path(__file__).parent.absolute(), 'data', 'model_input_data.csv')
        data = pd.read_csv(data_path)
        data.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')
    else:
        data = dt.create_basetable(df)
        data = fe.execute_feature_engineering_all(data)

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

    return data
