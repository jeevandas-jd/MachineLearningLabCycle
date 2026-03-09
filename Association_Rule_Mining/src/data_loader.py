import pandas as pd
import os

def load_transactions(file_path):

    df=pd.read_csv(file_path)

    item_set= df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).values.tolist


    return item_set


