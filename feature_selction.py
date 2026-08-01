import pandas as pd

def selction(df:pd.DataFrame):
    if 'equipment' in df.columns:
        df = pd.get_dummies(df, columns=['equipment'], drop_first=False)

    if 'pickup' in df.columns:
        df = df.drop(columns=['pickup'])

    if 'delivery' in df.columns:
        df = df.drop(columns=['delivery'])

    if 'load_id' in df.columns:
        df = df.drop(columns=['load_id'])

    print("feature selection completed successfully. Shape:", df.shape)
    return df
