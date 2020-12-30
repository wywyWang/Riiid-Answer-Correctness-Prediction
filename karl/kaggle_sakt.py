import gc
import psutil
import joblib
import random
from tqdm import tqdm

import numpy as np
import pandas as pd
import datatable as dt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import Dataset as DS
import Models as MD
#######################################
seed_value = 42
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value) # gpu vars
#######################################
TRAIN_SAMPLES = 320000
MAX_SEQ = 180
MIN_SAMPLES = 5
EMBED_DIM = 128
DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3
MAX_LEARNING_RATE = 2e-3
EPOCHS = 20
TRAIN_BATCH_SIZE = 64
#######################################
## Load Data
dtypes = {
    'timestamp': 'int64', 
    'user_id': 'int32' ,
    'content_id': 'int16',
    'content_type_id': 'int8',
    'answered_correctly':'int8'
}
train_df = dt.fread('data/train.csv', columns=set(dtypes.keys())).to_pandas()
for col, dtype in dtypes.items():
    train_df[col] = train_df[col].astype(dtype)
train_df = train_df[train_df.content_type_id == False]
train_df = train_df.sort_values(['timestamp'], ascending=True)
train_df.reset_index(drop=True, inplace=True)

## Preprocess
skills = train_df["content_id"].unique()
joblib.dump(skills, "skills.pkl.zip")
n_skill = len(skills)
print("number skills", len(skills))

group = train_df[['user_id', 'content_id', 'answered_correctly']].groupby('user_id')\
            .apply(lambda r: (r['content_id'].values, r['answered_correctly'].values))

joblib.dump(group, "group.pkl.zip")
del train_df
gc.collect()

print('preparing indexes and group')
train_indexes = list(group.index)[:TRAIN_SAMPLES]
valid_indexes = list(group.index)[TRAIN_SAMPLES:]
train_group = group[group.index.isin(train_indexes)]
valid_group = group[group.index.isin(valid_indexes)]
del group, train_indexes, valid_indexes
print(len(train_group), len(valid_group))

print('preparing training dataloader')
train_dataset = DS.SAKTDataset(train_group, n_skill, max_seq=MAX_SEQ)
train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=8)
del train_group

print('preparing validation dataloader')
valid_dataset = DS.SAKTDataset(valid_group, n_skill, max_seq=MAX_SEQ)
valid_dataloader = DataLoader(valid_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=8)
del valid_group
#############

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MD.SAKTModel(n_skill, max_seq=MAX_SEQ, embed_dim=EMBED_DIM, dropout_rate=DROPOUT_RATE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=MAX_LEARNING_RATE, steps_per_epoch=len(train_dataloader), epochs=EPOCHS
)

model.to(device)
criterion.to(device)

#############

best_auc = 0
max_steps = 3
step = 0
for epoch in range(EPOCHS):
    loss, acc, auc = MD.train_fn(model, train_dataloader, optimizer, scheduler, criterion, device)
    print("[epoch - {}/{}] [train: - {:.3f}] [acc - {:.3f}] [auc - {:.3f}]".format(epoch+1, EPOCHS, loss, acc, auc))
    loss, acc, auc = MD.valid_fn(model, valid_dataloader, criterion, device)
    print("[epoch - {}/{}] [valid: - {:.3f}] [acc - {:.3f}] [auc - {:.3f}]".format(epoch+1, EPOCHS, loss, acc, auc))
    if auc > best_auc:
        best_auc = auc
        step = 0
        torch.save(model.state_dict(), "sakt_model.pt")
    else:
        step += 1
        if step >= max_steps:
            break

del train_dataset, valid_dataset

torch.save(model.state_dict(), "sakt_model_final.pt")