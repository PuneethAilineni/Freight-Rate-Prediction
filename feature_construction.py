import pandas as pd

def construct_features(df:pd.DataFrame):
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month

    if 'distance' in df.columns and 'market_index' in df.columns:
        df['expected_cost_base'] = df['distance'] * df['market_index']

    if 'weight' in df.columns and 'distance' in df.columns:
        df['weight_per_mile'] = df['weight'] / df['distance']

    if 'quote_signal' in df.columns and 'distance' in df.columns:
        df['expected_quote_cost'] = df['quote_signal'] * df['distance']

    if 'pickup' in df.columns and 'posted_rate' in df.columns:
        pickup_mean = df.groupby('pickup')['posted_rate'].mean()
        df['pickup_mean_rate'] = df['pickup'].map(pickup_mean)

    if 'delivery' in df.columns and 'posted_rate' in df.columns:
        delivery_mean = df.groupby('delivery')['posted_rate'].mean()
        df['delivery_mean_rate'] = df['delivery'].map(delivery_mean)

    print("Feature construction completed successfully. Columns:", df.columns.tolist())
    return df
