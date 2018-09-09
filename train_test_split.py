from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

FILENAME_LOCATION = "./data/horizons.txt"
SEED = 0

df = pd.read_csv(FILENAME_LOCATION, sep="\s+", header=None, names=["filename", "width", "height", "horizon"], dtype={0: np.str})
# df = df[(df["width"] == 320) & (df["height"] == 240)]  # 531 pictures with exact dimensions

train_df, test_df = train_test_split(df, train_size=0.8, random_state=SEED)

train_df.to_csv("./data/train.txt", index=False)
test_df.to_csv("./data/test.txt", index=False)


