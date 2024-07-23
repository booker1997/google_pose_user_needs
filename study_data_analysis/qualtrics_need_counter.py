import pandas as pd
import numpy as np
import random
df = pd.read_csv('data/qualtrics_response_data.csv')
# print(df.columns[0])
need_counter_seen = {}
for i,col in enumerate(df.columns):
    if col not in need_counter_seen:
        need_counter_seen[col] = 0
    if i >= 19:
        response = df[col][2]
        if type(response) == str:
            need_counter_seen[col] += 1



