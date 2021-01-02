import gc
import psutil
import joblib
import random
import time
import os
from tqdm import tqdm
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import datatable as dt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import TruncatedSVD

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
torch.backends.cudnn.deterministic = True  #needed
torch.backends.cudnn.benchmark = False
#######################################
TRAIN_SAMPLES = 320000
MAX_SEQ = 180
MIN_SAMPLES = 5
EMBED_DIM = 128
DROPOUT_RATE = 0.2
LEARNING_RATE = 2e-3
MAX_LEARNING_RATE = 2e-3
EPOCHS = 20
TRAIN_BATCH_SIZE = 64
ACCEPTED_USER_CONTENT_SIZE = 1
TAGS_NUM = 188
MAX_TAGS_LEN = 6
#######################################

def compute_pretrained(data):
    cooccurrence_matrix = np.zeros((TAGS_NUM, TAGS_NUM), dtype=np.int)
    parsed_tags = []
    for tid, tags in enumerate(data):
        tags_list = [int(tag) for tag in tags.split(' ')]

        for tag_1, tag_2 in list(combinations_with_replacement(tags_list, 2)):
            # tag is 1-based
            if tag_1 == tag_2:
                cooccurrence_matrix[tag_1][tag_2] += 1
            else:
                cooccurrence_matrix[tag_1][tag_2] += 1
                cooccurrence_matrix[tag_2][tag_1] += 1

        # Pad to fixed length
        if len(tags_list) < MAX_TAGS_LEN:
            tags_list += [TAGS_NUM] * (MAX_TAGS_LEN - len(tags_list))
        else:
            assert Exception('ERROR: Tags Length greater than configure, please change the setting')
        parsed_tags.append(tags_list)

    # Use TSVD to get embedding
    svd = TruncatedSVD(n_components=EMBED_DIM, n_iter=7, random_state=seed_value)
    svd.fit(cooccurrence_matrix)
    padding_vector = np.zeros((1, EMBED_DIM))
    tags_embedding = np.concatenate([svd.components_.T, padding_vector], axis=0)

    return tags_embedding, parsed_tags


## Load Data

## Question csv
question_dtypes = { 
            'question_id': 'int16', 
            'part': 'int16'
         }

question_df = dt.fread('./riiid-test-answer-prediction/questions.csv', columns=set(question_dtypes.keys()).union({'tags'})).to_pandas()
for col, dtype in question_dtypes.items():
    question_df[col] = question_df[col].astype(dtype)

tags_embedding, parsed_tags = compute_pretrained(question_df['tags'].values)
question_df['tags'] = parsed_tags

## Train csv
train_dtypes = {
            'timestamp': 'int64', 
            'user_id': 'int32' , 
            'content_id': 'int16', 
            'content_type_id': 'int8', 
            'answered_correctly':'int8'
         }
train_df = dt.fread('./riiid-test-answer-prediction/train.csv', columns=set(train_dtypes.keys())).to_pandas()
for col, dtype in train_dtypes.items():
    train_df[col] = train_df[col].astype(dtype)
train_df = train_df[train_df.content_type_id == False]          # False means that a question was asked
train_df = train_df.sort_values(['timestamp'], ascending=True)
train_df.reset_index(drop=True, inplace=True)

## Create folder
folder = "model/"  + time.strftime("%Y%m%d-%H%M%S")
os.makedirs(folder, exist_ok=True)

## Preprocess
skills = train_df["content_id"].unique()
joblib.dump(skills, folder + "/skills.pkl.zip")
n_skill = len(skills)
print("number skills", len(skills))

group = train_df[['user_id', 'content_id', 'answered_correctly']].groupby('user_id').apply(lambda r: (
            r['content_id'].values,
            r['answered_correctly'].values))

joblib.dump(group, folder + "/group.pkl.zip")
del train_df
gc.collect()

train_indexes = list(group.index)[:TRAIN_SAMPLES]
valid_indexes = list(group.index)[TRAIN_SAMPLES:]
train_group = group[group.index.isin(train_indexes)]
valid_group = group[group.index.isin(valid_indexes)]
del group, train_indexes, valid_indexes
print(len(train_group), len(valid_group))

# Prepare training and validation data
train_dataset = DS.SAKTDataset(train_group, question_df, n_skill, max_seq=MAX_SEQ)
train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=8)
del train_group
valid_dataset = DS.SAKTDataset(valid_group, question_df, n_skill, max_seq=MAX_SEQ)
valid_dataloader = DataLoader(valid_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=8)
del valid_group

#############

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tags_embedding = torch.tensor(tags_embedding)
model = MD.SAKTModel(n_skill, max_seq=MAX_SEQ, embed_dim=EMBED_DIM, dropout=DROPOUT_RATE, enc_layers=1, pretrained_tags=tags_embedding)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=MAX_LEARNING_RATE, steps_per_epoch=len(train_dataloader), epochs=EPOCHS
)

model.to(device)
criterion.to(device)

# print(model)

#############

best_auc = 0
max_steps = 3
step = 0
for epoch in tqdm(range(EPOCHS)):
    loss, acc, auc = MD.train_fn(model, train_dataloader, optimizer, scheduler, criterion, device)
    print("[epoch - {}/{}] [train: - {:.3f}] [acc - {:.4f}] [auc - {:.4f}]".format(epoch+1, EPOCHS, loss, acc, auc))
    loss, acc, auc = MD.valid_fn(model, valid_dataloader, criterion, device)
    print("[epoch - {}/{}] [valid: - {:.3f}] [acc - {:.4f}] [auc - {:.4f}]\n".format(epoch+1, EPOCHS, loss, acc, auc))
    if auc > best_auc:
        best_auc = auc
        step = 0
        torch.save(model.state_dict(), folder + "/sakt_model.pt")
    else:
        step += 1
        if step >= max_steps:
            break
torch.save(model.state_dict(), folder + "/sakt_model_final.pt")