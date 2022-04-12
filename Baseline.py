import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, classification_report

data_train = pd.read_csv('LR_data_train.csv')
data_test = pd.read_csv('LR_data_test.csv')

def load_vocab(questions):
    vocab = {}
    for sent in questions:
        for word in sent.lower().split():
            vocab.setdefault(word, len(vocab))
    return vocab

feature_names = 'Question'
label_name = 'Round'

vocab = load_vocab(data_train['Question'].tolist())

X_train = np.zeros((len(data_train['Question'].tolist()), len(vocab)), dtype=int)
for index, feat_vec in enumerate(tqdm(X_train, desc='Load unigram feats')):
    for word in data_train['Question'].tolist():
        if word in vocab:
            word_idx = vocab[word]
            X_train[index][word_idx] += 1

X_test = np.zeros((len(data_test['Question'].tolist()), len(vocab)), dtype=int)
for index, feat_vec in enumerate(tqdm(X_test, desc='Load unigram feats')):
    for word in data_test['Question'].tolist():
        if word in vocab:
            word_idx = vocab[word]
            X_test[index][word_idx] += 1

y_train = np.array(data_train[label_name])

y_test = np.array(data_test[label_name])

# Fit and predict with the model
model = LogisticRegression()
model.fit(X_train, y_train)

y_train_pred =  [label for label in model.predict(X_train)]
y_test_pred = [label for label in model.predict(X_test)]

print(classification_report(y_train, y_train_pred))
print(classification_report(y_test, y_test_pred))
