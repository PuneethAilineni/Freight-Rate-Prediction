import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from data_cleaning import clean_data
from feature_construction import construct_features
from feature_selction import selction

def generate_all_predictions():
    train_df = pd.read_csv('data/train-test.csv')

    print("started prediction")
    median_market_index = train_df['market_index'].median()
    median_quote_signal = train_df['quote_signal'].median()
    
    lexington_lat = train_df[train_df['pickup'] == 'Lexington']['pickup_lat'].iloc[0]
    lexington_lon = train_df[train_df['pickup'] == 'Lexington']['pickup_lon'].iloc[0]
    fort_wayne_lat = train_df[train_df['delivery'] == 'Fort Wayne']['delivery_lat'].iloc[0]
    fort_wayne_lon = train_df[train_df['delivery'] == 'Fort Wayne']['delivery_lon'].iloc[0]

    train_df = clean_data(train_df)
    train_df = construct_features(train_df)
    train_df = selction(train_df)

    features = [col for col in train_df.columns if col not in ['date', 'posted_rate']]

    X_train = train_df[features]
    y_train_rpm = train_df['posted_rate'] / train_df['distance']
    y_train_rpm_log = np.log1p(y_train_rpm)

    try:
        model = HistGradientBoostingRegressor(
            max_iter=500, max_depth=6, learning_rate=0.05, l2_regularization=1.0, random_state=42
        )
        model.fit(X_train, y_train_rpm_log)
    except Exception as e:
        print("something went wrong at training => {e}")
    print("training completed")

    print("\n--- 3. Processing and Predicting: validation.csv ---")
    val_df = pd.read_csv('data/validation.csv')
    
    load_ids = val_df['load_id'].copy()

    val_df = clean_data(val_df)
    val_df = construct_features(val_df)
    val_df = selction(val_df)

    for col in features:
        if col not in val_df.columns:
            val_df[col] = 0

    X_val = val_df[features]
    preds_rpm_log_val = model.predict(X_val)
    
    preds_rpm_val = np.expm1(preds_rpm_log_val)
    final_preds_val = preds_rpm_val * val_df['distance']

    submission = pd.DataFrame({
        'load_id': load_ids,
        'predicted_rate': final_preds_val
    })
    val_output_path = 'validation_predictions.csv'
    submission.to_csv(val_output_path, index=False)
    print(f"Saved {len(submission)} predictions to {val_output_path}")

    dec_file = 'data/december_chart_inputs.csv'
    dec_df = pd.read_csv(dec_file)
    
    original_dec_columns = list(dec_df.columns)

    dec_df['pickup_lat'] = lexington_lat
    dec_df['pickup_lon'] = lexington_lon
    dec_df['delivery_lat'] = fort_wayne_lat
    dec_df['delivery_lon'] = fort_wayne_lon
    dec_df['market_index'] = median_market_index
    dec_df['quote_signal'] = median_quote_signal

    dec_processed = clean_data(dec_df.copy())
    dec_processed = construct_features(dec_processed)
    dec_processed = selction(dec_processed)
    
    for col in features:
        if col not in dec_processed.columns:
            dec_processed[col] = False

    X_dec = dec_processed[features]
    preds_rpm_log_dec = model.predict(X_dec)
    final_preds_dec = np.expm1(preds_rpm_log_dec) * dec_df['distance']

    dec_df['predicted_rate'] = final_preds_dec
    dec_df = dec_df[original_dec_columns]
    
    dec_df.to_csv(dec_file, index=False)
    print("prediction completed")

if __name__ == '__main__':
    generate_all_predictions()