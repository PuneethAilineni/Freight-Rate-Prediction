import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def training(df:pd.DataFrame):
    train_df = df[df['month'] <= 8].copy()
    test_df = df[df['month'] >= 9].copy()
    y_train = train_df['posted_rate']
    y_test = test_df['posted_rate']

    X_train = train_df.drop(columns=['posted_rate', 'date'], errors='ignore')
    X_test = test_df.drop(columns=['posted_rate', 'date'], errors='ignore')

    y_train_rpm = y_train / X_train['distance']

    y_train_rpm_log = np.log1p(y_train_rpm)

    model = HistGradientBoostingRegressor(
        max_iter=500, max_depth=6, learning_rate=0.05, l2_regularization=1.0, random_state=42
    )
    model.fit(X_train, y_train_rpm_log)

    preds_rpm_log = model.predict(X_test)
    preds_rpm = np.expm1(preds_rpm_log)
    final_preds = preds_rpm * test_df['distance']

    mae = mean_absolute_error(y_test, final_preds)
    rmse = np.sqrt(mean_squared_error(y_test, final_preds))

    print(f"MAE: ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")