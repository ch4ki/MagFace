import pandas as pd

data = pd.read_csv("db_files/faiss.csv",header=None)

data.columns = ["path", "encodings"]

sample = list(eval(data["encodings"].iloc[0]))