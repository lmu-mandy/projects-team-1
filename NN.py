import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, classification_report


data_train = pd.read_csv('LR_data_train.csv')
data_test = pd.read_csv('LR_data_test.csv')


questions = list(data_train['Question'])
rounds = list(data_train['Round'])
train_data = list(zip(questions,rounds))

questionsTest = list(data_train['Question'])
roundsTest = list(data_train['Round'])
test_data = list(zip(questionsTest,roundsTest))

#print(train_data)
# Build an LSTM model for tagging:

class LSTM(nn.Module):
  def __init__(self, num_words, emb_dim, num_y, hidden_dim=32):
    super().__init__()
    self.emb_dim = emb_dim
    self.emb = nn.Embedding(num_words, emb_dim)
    self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=1, bidirectional=False)
    self.linear = nn.Linear(hidden_dim, num_y)
    self.softmax = nn.LogSoftmax(dim=0)
    #self.embed_bag = nn.EmbeddingBag(num_words, num_y, mode='mean')

  def forward(self, text):
    embeds = self.emb(text)
    #embeds = self.embed_bag(text)
    # print(embeds.shape)
    # embeds = torch.mean(embeds)
    # print(embeds.shape)

    # LSTM input size: (sequence length, batch size, hidden_dim)
    #print(embeds.shape)
    out, (last_hidden, last_cell) = self.lstm(embeds.view( 1, len(text), -1))

    # Keep all hidden states for tagging
    tag_space = self.linear(out.view(len(text), -1)) 
    return self.softmax(tag_space)

# Define a function for loading the vocabulary of tokens, as well as mapping to and from tags.

def load_vocab_tags(train_data):
  word_to_ix = {} # Maps tokens to an index in the vocabulary.
  tag_to_ix = {} # Maps tag labels to a unique tag index.
  ix_to_tag = {} # Maps tag indices back to tag strings.
  for sent, tags in train_data:
    for word in sent.split():
      word_to_ix.setdefault(word, len(word_to_ix))
    tag_to_ix.setdefault(tags, len(tag_to_ix))
    ix_to_tag[tag_to_ix[tags]] = tags
  return word_to_ix, tag_to_ix, ix_to_tag

# Define sample sentences with tags for each token (O: Other, BE: Begin-Emotion, IE: Inside-Emotion):

tok_to_ix, tag_to_ix, ix_to_tag = load_vocab_tags(train_data)

# Create a new model and set the optimizer and loss function:

emb_dim = 50
learning_rate = 0.01
model = LSTM(len(tok_to_ix), emb_dim, len(tag_to_ix))
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()

# Train the model for however many epochs you want!

n_epochs = 10
for epoch in range(n_epochs):
  model.train()
  for text, tags in train_data:
    x = [tok_to_ix[tok] for tok in text.split()]
    y = [tag_to_ix[tags] for i in range(len(x))]
    x_train_tensor = torch.LongTensor(x)
    y_train_tensor = torch.LongTensor(y)
    pred_y = model(x_train_tensor)#.view(1, x_train_tensor.shape[0]))
    loss = loss_fn(pred_y, y_train_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
  print("\nEpoch:", epoch)
  print("Training loss:", loss.item())

# Last step! Evaluate the model to see if it predicts the right label:

true_y = []
pred_y = []
x_test = []
for sentence, label in test_data:
    with torch.no_grad():
        model.eval()
        true_y.append(label)
        x = []
        for word in sentence.split():
            if word in tok_to_ix:
                x += [tok_to_ix[word]]
        if (len(x) > 0):
            x_test = torch.LongTensor(x)
            pred_y_test = model(x_test)
            output_tags = [ix_to_tag[max_ix] for max_ix in pred_y_test.argmax(1).data.numpy()]
            [print(output_tags)]
            predict = min(set(output_tags), key = output_tags.count)
            pred_y.append(predict)

report = classification_report(true_y, pred_y)
print(report)

