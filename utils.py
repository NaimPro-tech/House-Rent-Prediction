import pandas as pd

def cv_target_encoding(X, y, cols, n_split=5):
    import numpy as np
    from category_encoders import TargetEncoder
    from sklearn.model_selection import KFold

    kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)

    X_enc = X.copy()

    for col in cols:
        X_enc[col] = np.nan #at first we make them nan to put our encoded value

        for train_idx, val_idx in kfold.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = y.iloc[train_idx]

            te = TargetEncoder(cols=[col], smoothing=10)

            #mapping learn--only in fold train
            te.fit(X_tr[[col]], y_tr)

            #apply on fol val
            X_enc.iloc[val_idx, X.columns.get_loc(col)] = te.transform(X_val[col])[col]

    return X_enc


def create_features(df):
    df['floor_ratio'] = df['living_floor']/df['total_floor'].replace(0,1)
    df['size_per_bhk'] = df['size']/df['bhk'].replace(0,1)
    df['bathroom_per_room'] = df['bathroom']/df['bhk'].replace(0,1)

    return df


def preprocess(df, encoder, features):
    df = create_features(df)
    df['area_locality'] = encoder.transform(df['area_locality'])
    df = pd.get_dummies(df)
    df = df.reindex(columns=features, fill_value=0)
    return df