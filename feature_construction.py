import pandas as pd

def construct_features(df:pd.DataFrame):
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month

    if 'distance' and 'market_index' in df.columns:
        df['expected_cost_base'] = df['distance'] * df['market_index']

    if 'weight' and 'distance' in df.columns:
        df['weight_per_mile'] = df['weight'] / df['distance']

    if 'quote_signal' and 'distance' in df.columns:
        df['expected_quote_cost'] = df['quote_signal'] * df['distance']

    print("Feature construction completed successfully. Columns:", df.columns.tolist())
    return df
