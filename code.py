import numpy as np
import pandas as pd
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertModel, DistilBertTokenizer
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from torch import cuda
import random

import logging
logging.basicConfig(level=logging.ERROR)

device = 'cuda' if cuda.is_available() else 'cpu'
print(cuda.is_available())

dataDF = pd.read_csv('./data/data.csv', sep=',')
dataDF.dropna(inplace=True)

tmp = dataDF.values.tolist()

rootDir='./data/'
l = []
for i in tmp:
  lTmp = []
  lTmp.append(i[0])
  fName = i[1][:-4]
  fName = rootDir + fName
  fp = open(fName)
  des = fp.read()
  i[1] = des
  lTmp.append(i[1])
  lTmp.append(i[2:])
  l.append(lTmp)

random.shuffle(l)

sz = len(l)
train = l[:int(sz*0.8)]
test = l[int(sz*0.8):]

print(len(train))
print(len(test))

col_name = ['no', 'text', 'labels']
train_df = pd.DataFrame(train, columns=col_name)
test_df = pd.DataFrame(test, columns=col_name)

print(train_df)
print(test_df)

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 70
LEARNING_RATE = 1e-04

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
        
training_set = MultiLabelDataset(train_df, tokenizer, MAX_LEN)
testing_set = MultiLabelDataset(test_df, tokenizer, MAX_LEN)
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 38)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

model = DistilBERTClass()
model.to(device)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

def train_model(epoch):
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)
        #outputs = model(ids, mask)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _%1000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
for epoch in range(EPOCHS):
    train_model(epoch)


def validation(testing_loader):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets
outputs, targets = validation(testing_loader)
outputs = np.array(outputs) >= 0.5
targets = np.array(targets) == True
accuracy = metrics.accuracy_score(targets, outputs)f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
f1_score_macro = metrics.f1_score(targets, outputs, average='macro')

res = 0
for i in range(len(targets)):
  cnt = 0
  for j in range(len(targets[i])):
    cnt = cnt + (1 if targets[i][j] == outputs[i][j] else 0)
  res = res + (1 if cnt > 0 else 0)
print(str(res) + "/" + str(len(targets)))
print(f"Accuracy Score = {accuracy}")
print(f"F1 Score (Micro) = {f1_score_micro}")
print(f"F1 Score (Macro) = {f1_score_macro}")