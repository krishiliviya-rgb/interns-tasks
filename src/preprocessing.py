import pandas as pd

def load_and_preprocess(filepath):
    """
    Load dataset and perform preprocessing.
    """

    df = pd.read_csv(filepath)

    # Convert Date column
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort by date
    df = df.sort_values("Date")

    # Aggregate sales per day
    df = df.groupby('Date')['Sales'].sum().reset_index()

    # Feature Engineering
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    return df