import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from category_encoders import TargetEncoder
from necessary_functions import cv_target_encoding
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("House_Rent_Dataset.csv")

df.columns = [col.lower().replace(" ", "_") for col in df.columns]

df['rent'] = np.log1p(df['rent'])

#outlier capping with IQR
q1 = df['rent'].quantile(0.25)
q3 = df['rent'].quantile(0.75)

iqr = q3-q1

upper = q3+1.5*iqr
lower = q1-1.5*iqr

df['rent'] = df['rent'].clip(upper, lower)


df['floor'] = df['floor'].str.replace("Ground", "0")

df[['living_floor', 'total_floor']] = df['floor'].str.split(' out of ', expand=True)

df['living_floor'] = df['living_floor'].str.replace({
    'Lower Basement':'-2',
    'Upper Basement':'-1'
})

df.dropna(subset=['total_floor'], inplace=True)

df['living_floor'] = df['living_floor'].astype(int)
df['total_floor'] = df['total_floor'].astype(int)

df['floor_ratio'] = df['living_floor']/df['total_floor']
df['size_per_bhk'] = df['size']/df['bhk']
df['bathroom_per_room'] = df['bathroom']/df['bhk']


X = df.drop('rent', axis=1)
y = df['rent']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_temp, X_test, y_temp, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

df['is_bachelors_allowed'] = df['tenant_preferred'].apply(lambda x: 1 if 'Bachelors' in x else 0)
df['is_family_allowed'] = df['tenant_preferred'].apply(lambda x: 1 if 'Family' in x else 0)

cols = ['area_locality']

X_temp_enc = X_temp.copy()
X_test_enc = X_test.copy()

X_train_enc = cv_target_encoding(X_train, y_train, cols)

te_full = TargetEncoder(cols=cols, smoothing=10)
te_full.fit(X_train[cols], y_train)

X_temp_enc[cols] = te_full.transform(X_temp[cols])
X_test_enc[cols] = te_full.transform(X_test[cols])

X_train_enc = pd.get_dummies(X_train_enc, columns=['city', 'area_type'], drop_first=True)
X_temp_enc = pd.get_dummies(X_temp_enc, columns=['city', 'area_type'], drop_first=True)
X_test_enc = pd.get_dummies(X_test_enc, columns=['city', 'area_type'], drop_first=True)


X_temp_enc = X_temp_enc.reindex(columns=X_train_enc.columns, fill_value=0)
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

feature_columns = X_train_enc.select_dtypes(include=['int64','float64', 'bool']).columns

param_dist = {
    'n_estimators':[200,300,400,500],
    'max_depth':[4,5,6,7],
    'learning_rate':[0.01, 0.05, 0.1],
    'subsample':[0.7,0.8,0.9],
    'colsample_bytree':[0.7,0.8,0.9]
}

xgb = XGBRegressor(random_state=42)

random_search = RandomizedSearchCV(
    xgb,
    param_distributions=param_dist,
    cv=3,
    n_iter=20,
    scoring='r2',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train_enc[feature_columns], y_train)

best_xgb = random_search.best_estimator_

y_val_xgb = best_xgb.predict(X_temp_enc[feature_columns])
y_test_xgb = best_xgb.predict(X_test_enc[feature_columns])

print(f"Validation Score(XGB with Random Search CV): {mean_squared_error(y_temp, y_val_xgb)}")
print(f"Validation MAE: {mean_absolute_error(y_temp, y_val_xgb)}")
print(f"Validation R2: {r2_score(y_temp, y_val_xgb)}")
print(f"Test Score(XGB with Random Search CV): {mean_squared_error(y_test, y_test_xgb)}")
print(f"Test MAE: {mean_absolute_error(y_test, y_test_xgb)}")
print(f"Test R2: {r2_score(y_test, y_test_xgb)}")

y_test_actual = np.expm1(y_test_xgb)


#model save
joblib.dump(best_xgb, "house_rent_prediction_model.pkl")

#feature list save
joblib.dump(feature_columns.tolist(), "feature_columns.pkl")

#save all the necessary model artifacts
joblib.dump({
    "model": best_xgb,
    "target_encoder": te_full,
    "features": feature_columns.tolist()
}, "artifacts.pkl")

data = joblib.load("artifacts.pkl")
print(data.keys())