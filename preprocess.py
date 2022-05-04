import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

def value_preprocessing(value):
    if value == 'None':
        value = 0
    elif value[0] == '$':
        value = str(value[1:len(value)])
    return value

columns_to_keep = [' Round', ' Value', ' Question']
data = pd.read_csv('JEOPARDY_CSV.csv')

data2 = data[columns_to_keep].copy()
data2.columns = ['Round', 'Value', 'Question']
data2['Value'] = data2['Value'].apply(value_preprocessing)

ronud_to_idx = {'Jeopardy!':0, 'Double Jeopardy!':1, 'Final Jeopardy!':2}
data2 = data2.replace(ronud_to_idx)
print(data2)

halved = data2.sample(frac=.15,axis=0) # edit the frac keyword to adjust size of the dataset you use. 20% uses about 30gb of data to allocate the arrays
data_test = halved.sample(frac=.2, axis=0)
data_train = halved.drop(index=data_test.index)


data_train.to_csv('LR_data_train.csv', index=False)
data_test.to_csv('LR_data_test.csv', index=False)

