import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df['weight'] = df['weight'].abs().fillna(df['weight'].median())
    df['market_index'] = df['market_index'].fillna(df['market_index'].median())
    print("Null counts:")
    print(df[['weight', 'market_index']].isnull().sum())

    return df
