import pandas as pd

def selction(df:pd.DataFrame):
    if 'equipment' in df.columns:
        df = pd.get_dummies(df, columns=['equipment'], drop_first=False)

    if 'load_id' in df.columns:
        df = df.drop(columns=['load_id'])

    for col in ['pickup', 'delivery']:
        if col in df.columns:
            df = df.drop(columns=[col])

    print("feature selection completed successfully. Shape:", df.shape)
    return df
