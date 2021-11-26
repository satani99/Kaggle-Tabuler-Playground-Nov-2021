import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

torch.cuda.empty_cache()
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.drop(["id","f0", "f6", "f12", "f13", "f18", "f29", "f38", "f39", "f46", "f52", "f59", "f63", "f65", "f72", "f73", "f74", "f79", "f85", "f86", "f87", "f89", "f92", "f93"], axis=1, inplace=True)
test.drop(["id", "f0", "f6", "f12", "f13", "f18", "f29", "f38", "f39", "f46", "f52", "f59", "f63", "f65", "f72", "f73", "f74", "f79", "f85", "f86", "f87", "f89", "f92", "f93"], axis=True, inplace=True)

y, X = train['target'], train.drop(['target'], axis=1)

print(X.head)
print(y.head)



sns.countplot(x = 'target', data = train)
plt.show()
print('plot showed')

scaler = StandardScaler()
X = scaler.fit_transform(X)
test = scaler.fit_transform(test)
print('input is standardized')

EPOCHS = 50
BATCH_SIZE = 15000
LEARNING_RATE = 0.001

class TrainData(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

train_data = TrainData(torch.FloatTensor(X), torch.FloatTensor(y))

class TestData(Dataset):

    def __init__(self, X):
        self.X = X

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return len(self.X)

test_data = TestData(torch.FloatTensor(test))


train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

print('train_loader completed')

test_loader = DataLoader(dataset=test_data, batch_size=1)

print('test_loader completed')

class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()

        self.layer_1 = nn.Linear(X.shape[1], 100)
        self.layer_2 = nn.Linear(100, 100)
        self.layer_out = nn.Linear(100, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.batchnorm1 = nn.BatchNorm1d(100)
        self.batchnorm2 = nn.BatchNorm1d(100)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = BinaryClassification()
model.to(device)

print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

model.train()
for e in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        y_pred = model(X_batch)

        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

y_pred_list = []

model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())


y_pred_list = [a.squeeze().tolist() for a in y_pred_list]










sub = pd.read_csv('sample_submission.csv')
sub['target'] = y_pred_list
print(sub)
sub.to_csv('submission.csv', index=False)


