import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

col = ['userId', 'movieId', 'rating', 'timestamp']

data = pd.read_csv('ml-1m/ratings.dat', sep='::', names=col)

train, test = train_test_split(data, test_size=0.2)
print(train)
train.sort_values('userId').to_csv('./data-split/train.csv', index=False)
test.sort_values('userId').to_csv('./data-split/test.csv', index=False)

