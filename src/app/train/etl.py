import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    # ejemplo de limpieza
    df = df.drop_duplicates()
    df = df.dropna(subset=["columna_importante"])
    return df

def transform_data(df):
    # ejemplo de transformación
    df["nueva_variable"] = df["columna1"] / df["columna2"]
    return df

def run_etl(path):
    df = load_data(path)
    df = clean_data(df)
    df = transform_data(df)
    return df

if __name__ == "__main__":
    df_final = run_etl("data/tu_dataset.csv")
    print(df_final.head())
