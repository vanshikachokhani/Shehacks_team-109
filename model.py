import numpy as np
import pandas as pd
import pickle

features = ['smart_5_raw', 'smart_187_raw', 'smart_188_raw',
            'smart_197_raw', 'smart_198_raw', 'failure']
train_data = pd.read_csv("./dataset/jan_feb_backblaze_train.csv").reindex(columns=features)
test_data = pd.read_csv("./dataset/nov_dec_backblaze_test.csv").reindex(columns=features)

train_data = train_data.fillna(value=-1)
test_data = test_data.fillna(value=-1)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

train_ds = train_data.drop(['failure'], axis=1)
train_target = train_data['failure']

test_ds = test_data.drop(['failure'], axis=1)
test_target = test_data['failure']

gnb = gnb.fit(train_ds, train_target)

with open("model.pkl", 'wb') as f_out:
    pickle.dump(gnb, f_out)
    f_out.close()
with open('model.pkl', 'rb') as f_in:
    model = pickle.load(f_in)