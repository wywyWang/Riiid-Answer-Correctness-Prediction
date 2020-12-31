import psutil
import joblib

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import riiideducation

env = riiideducation.make_env()
iter_test = env.iter_test()

TRAIN_SAMPLES = 320000
MAX_SEQ = 200
MIN_SAMPLES = 5
EMBED_DIM = 128
DROPOUT_RATE = 0.2
ACCEPTED_USER_CONTENT_SIZE = 4
WEIGHT_PTH = '/kaggle/input/weight-and-data/sakt_model_final.pt'

##################### MODEL ##############################
class FFN(nn.Module):
    def __init__(self, state_size=200, forward_expansion=1, bn_size=MAX_SEQ-1, dropout=0.2):
        super(FFN, self).__init__()
        self.state_size = state_size
        
        self.lr1 = nn.Linear(state_size, forward_expansion * state_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(bn_size)
        self.lr2 = nn.Linear(forward_expansion * state_size, state_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.relu(self.lr1(x))
        x = self.bn(x)
        x = self.lr2(x)
        return self.dropout(x)

def future_mask(seq_length):
    future_mask = (np.triu(np.ones([seq_length, seq_length]), k = 1)).astype('bool')
    return torch.from_numpy(future_mask)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads=8, dropout=DROPOUT_RATE, forward_expansion=1):
        super(TransformerBlock, self).__init__()
        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_normal = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, forward_expansion = forward_expansion, dropout=dropout)
        self.layer_normal_2 = nn.LayerNorm(embed_dim)
        

    def forward(self, value, key, query, att_mask):
        att_output, att_weight = self.multi_att(value, key, query, attn_mask=att_mask)
        att_output = self.dropout(self.layer_normal(att_output + value))
        att_output = att_output.permute(1, 0, 2) # att_output: [s_len, bs, embed] => [bs, s_len, embed]
        x = self.ffn(att_output)
        x = self.dropout(self.layer_normal_2(x + att_output))
        return x.squeeze(-1), att_weight
    
class Encoder(nn.Module):
    def __init__(self, n_skill, max_seq=100, embed_dim=128, dropout=DROPOUT_RATE, forward_expansion=1, num_layers=1, heads = 8):
        super(Encoder, self).__init__()
        self.n_skill, self.embed_dim = n_skill, embed_dim
        self.embedding = nn.Embedding(2 * n_skill + 1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq - 1, embed_dim)
        self.e_embedding = nn.Embedding(n_skill+1, embed_dim)
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, forward_expansion = forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, question_ids):
        device = x.device
        x = self.embedding(x)
        pos_id = torch.arange(x.size(1)).unsqueeze(0).to(device)
        pos_x = self.pos_embedding(pos_id)
        x = self.dropout(x + pos_x)
        x = x.permute(1, 0, 2) # x: [bs, s_len, embed] => [s_len, bs, embed]
        e = self.e_embedding(question_ids)
        e = e.permute(1, 0, 2)
        for layer in self.layers:
            att_mask = future_mask(e.size(0)).to(device)
            x, att_weight = layer(e, x, x, att_mask=att_mask)
            x = x.permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        return x, att_weight

class SAKTModel(nn.Module):
    def __init__(self, n_skill, max_seq=100, embed_dim=128, dropout=DROPOUT_RATE, forward_expansion = 1, enc_layers=1, heads = 8):
        super(SAKTModel, self).__init__()
        self.encoder = Encoder(n_skill, max_seq, embed_dim, dropout, forward_expansion, num_layers=enc_layers)
        self.pred = nn.Linear(embed_dim, 1)
        
    def forward(self, x, question_ids):
        x, att_weight = self.encoder(x, question_ids)
        x = self.pred(x)
        return x.squeeze(-1), att_weight    
##################################################################

print('loading skill pkl')
skills = joblib.load("/kaggle/input/weight-and-data/skills.pkl.zip")
n_skill = len(skills)
print('loading graoup pkl')
group = joblib.load("/kaggle/input/weight-and-data/group.pkl.zip")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device = {device}')
model = SAKTModel(n_skill, max_seq=MAX_SEQ, embed_dim=EMBED_DIM, dropout=DROPOUT_RATE)
try:
    print('load model weight')
    model.load_state_dict(torch.load(WEIGHT_PTH))
except:
    print('ERROR: change to load model weight map location at cpu')
    model.load_state_dict(torch.load(WEIGHT_PTH, map_location='cpu'))
model.to(device)
model.eval()

##################################

class TestDataset(Dataset):
    def __init__(self, samples, test_df, n_skill, max_seq=100):
        super(TestDataset, self).__init__()
        self.samples, self.user_ids, self.test_df = samples, [x for x in test_df["user_id"].unique()], test_df
        self.n_skill, self.max_seq = n_skill, max_seq

    def __len__(self):
        return self.test_df.shape[0]
    
    def __getitem__(self, index):
        test_info = self.test_df.iloc[index]
        
        user_id = test_info['user_id']
        target_id = test_info['content_id']
        
        content_id_seq = np.zeros(self.max_seq, dtype=int)
        answered_correctly_seq = np.zeros(self.max_seq, dtype=int)
        
        if user_id in self.samples.index:
            content_id, answered_correctly = self.samples[user_id]
            
            seq_len = len(content_id)
            
            if seq_len >= self.max_seq:
                content_id_seq = content_id[-self.max_seq:]
                answered_correctly_seq = answered_correctly[-self.max_seq:]
            else:
                content_id_seq[-seq_len:] = content_id
                answered_correctly_seq[-seq_len:] = answered_correctly
                
        x = content_id_seq[1:].copy()
        x += (answered_correctly_seq[1:] == 1) * self.n_skill
        
        questions = np.append(content_id_seq[2:], [target_id])
        
        return x, questions

##################################

prev_test_df = None
print('start testing')

for (test_df, sample_prediction_df) in tqdm(iter_test):
    
    if (prev_test_df is not None) & (psutil.virtual_memory().percent<90):
        print(psutil.virtual_memory().percent)
        prev_test_df['answered_correctly'] = eval(test_df['prior_group_answers_correct'].iloc[0])
        prev_test_df = prev_test_df[prev_test_df.content_type_id == False]
        prev_group = prev_test_df[['user_id', 'content_id', 'answered_correctly']].groupby('user_id').apply(lambda r: (
            r['content_id'].values,
            r['answered_correctly'].values))
        for prev_user_id in prev_group.index:
            prev_group_content = prev_group[prev_user_id][0]
            prev_group_answered_correctly = prev_group[prev_user_id][1]
            if prev_user_id in group.index:
                group[prev_user_id] = (np.append(group[prev_user_id][0], prev_group_content), 
                                       np.append(group[prev_user_id][1], prev_group_answered_correctly))
            else:
                group[prev_user_id] = (prev_group_content, prev_group_answered_correctly)
            
            if len(group[prev_user_id][0]) > MAX_SEQ:
                new_group_content = group[prev_user_id][0][-MAX_SEQ:]
                new_group_answered_correctly = group[prev_user_id][1][-MAX_SEQ:]
                group[prev_user_id] = (new_group_content, new_group_answered_correctly)
                
    prev_test_df = test_df.copy()
    test_df = test_df[test_df.content_type_id == False]
    
    test_dataset = TestDataset(group, test_df, n_skill, max_seq=MAX_SEQ)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_df), shuffle=False)
    
    item = next(iter(test_dataloader))
    x = item[0].to(device).long()
    target_id = item[1].to(device).long()
    
    with torch.no_grad():
        output, _ = model(x, target_id)
        
    output = torch.sigmoid(output)
    output = output[:, -1]
    test_df['answered_correctly'] = output.cpu().numpy()
    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])

print('finish')