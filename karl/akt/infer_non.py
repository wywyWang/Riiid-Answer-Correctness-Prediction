import psutil

import pandas as pd
import sys
import os
import os.path
import glob
import logging
import argparse
import numpy as np
import torch
import datatable as dt
import joblib
import gc

from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import math
import torch.nn.functional as F
from enum import IntEnum
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import riiideducation

env = riiideducation.make_env()
iter_test = env.iter_test()

WEIGHT_PTH = '../input/akt-no-concept/output_12'

parser = argparse.ArgumentParser(description='Script to test KT')
# Basic Parameters
parser.add_argument('--max_iter', type=int, default=300,
                    help='number of iterations')
parser.add_argument('--train_set', type=int, default=1)
parser.add_argument('--seed', type=int, default=224, help='default seed')

# Common parameters
parser.add_argument('--optim', type=str, default='adam',
                    help='Default Optimizer')
parser.add_argument('--batch_size', type=int,
                    default=128, help='the batch size')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--maxgradnorm', type=float,
                    default=-1, help='maximum gradient norm')
parser.add_argument('--final_fc_dim', type=int, default=512,
                    help='hidden state dim for final fc layer')

# AKT Specific Parameter
parser.add_argument('--d_model', type=int, default=256,
                    help='Transformer d_model shape')
parser.add_argument('--d_ff', type=int, default=1024,
                    help='Transformer d_ff shape')
parser.add_argument('--dropout', type=float,
                    default=0.05, help='Dropout rate')
parser.add_argument('--n_block', type=int, default=1,
                    help='number of blocks')
parser.add_argument('--n_head', type=int, default=8,
                    help='number of heads in multihead attention')
parser.add_argument('--kq_same', type=int, default=1)

# AKT-R Specific Parameter
parser.add_argument('--l2', type=float,
                    default=1e-5, help='l2 penalty for difficulty')

parser.add_argument('--max_seq', type=int, default=200)
parser.add_argument('--file_name', type=str, default='output')

params = parser.parse_args([])
##################### MODEL ##############################


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class AKT(nn.Module):
    def __init__(self, n_question, d_model, n_blocks,
                 kq_same, dropout, final_fc_dim=512, n_heads=8, d_ff=2048,  l2=1e-5, separate_qa=False):
        super().__init__()
        self.user_concept = False
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        # self.n_pid = n_pid
        self.n_pid = -1
        self.l2 = l2
        self.separate_qa = separate_qa
        embed_l = d_model
        if self.user_concept:
            # self.difficult_param = nn.Embedding(self.n_pid+1, 1)
            # self.difficult_part = nn.Embedding(8, 1)
            # self.difficult_tag = nn.Embedding(189, 1, padding_idx=188)
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)
        # n_question+1 ,d_model
        self.q_embed = nn.Embedding(self.n_question+1, embed_l)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
        else:
            self.qa_embed = nn.Embedding(2, embed_l)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / n_heads, d_ff=d_ff,  kq_same=self.kq_same)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        # self.reset()

    # def reset(self):
    #     for p in self.parameters():
    #         if p.size(0) == self.n_pid+1 and self.n_pid > 0:
    #             torch.nn.init.constant_(p, 0.)

    def predict(self, q_data, qa_data):
        # Batch First
        q_embed_data = self.q_embed(q_data)
        if self.separate_qa:
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_data = (qa_data-q_data)//self.n_question
            qa_embed_data = self.qa_embed(qa_data)+q_embed_data

        if self.user_concept:
            q_embed_diff_data = self.q_embed_diff(q_data)
            # pid_part = self.difficult_part(part_seq)
            # pid_tags = self.difficult_tag(tags_seq)
            # pid_tags = torch.mean(pid_tags, dim=2)
            # pid_embed_data = pid_part + pid_tags    # uq

            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data  # uq *d_ct + c_ct
            qa_embed_diff_data = self.qa_embed_diff(qa_data)  # f_(ct,rt) or #h_rt
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data+q_embed_diff_data)  # + uq *(h_rt+d_ct)


        d_output = self.model(q_embed_data, qa_embed_data)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q)

        return output


class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model

        self.blocks_1 = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
            for _ in range(n_blocks)
        ])
        self.blocks_2 = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
            for _ in range(n_blocks*2)
        ])

    def forward(self, q_embed_data, qa_embed_data):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        for block in self.blocks_1:  # encode qas
            y = block(mask=1, query=y, key=y, values=y)
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False)
                flag_first = False
            else:  # dont peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
                flag_first = True
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True)
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        gammas = self.gammas
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad, gammas)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
        scores_ = scores_ * mask.float().to(device)
        distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1
        position_effect = torch.abs(
            x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen
        # bs, 8, sl, sl positive distance
        dist_scores = torch.clamp(
            (disttotal_scores-distcum_scores)*position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    total_effect = torch.clamp(torch.clamp(
        (dist_scores*gamma).exp(), min=1e-5), max=1e5)
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


##################################################################

print('loading skill pkl')
skills = joblib.load("/kaggle/input/weight-and-data/skills.pkl.zip")
n_skill = len(skills)
params.n_question = n_skill
print('loading graoup pkl')
group = joblib.load("/kaggle/input/weight-and-data/group.pkl.zip")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device = {device}')
model = AKT(n_question=params.n_question, n_blocks=params.n_block, d_model=params.d_model,
            dropout=params.dropout, kq_same=params.kq_same, l2=params.l2, n_heads=params.n_head,
            d_ff=params.d_ff, final_fc_dim=params.final_fc_dim).to(device)
try:
    print('load model weight')
    checkpoint = torch.load(WEIGHT_PTH)
    model.load_state_dict(checkpoint['model_state_dict'])
except:
    print('ERROR: change to load model weight map location at cpu')
    checkpoint = torch.load(WEIGHT_PTH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

##################################
class TestDataset(Dataset):
    def __init__(self, samples, test_df, n_skill, max_seq):
        super(TestDataset, self).__init__()
        self.samples = samples
        self.user_ids = [x for x in test_df["user_id"].unique()]
        self.test_df = test_df
        self.n_skill = n_skill
        self.max_seq = max_seq

    def __len__(self):
        return self.test_df.shape[0]

    def __getitem__(self, index):
        test_info = self.test_df.iloc[index]

        user_id = test_info["user_id"]
        target_id = test_info["content_id"]

        content_id_seq = np.zeros(self.max_seq, dtype=int)
        answered_correctly_seq = np.zeros(self.max_seq, dtype=int)

        if user_id in self.samples.index:
            content_id, answered_correctly = self.samples[user_id]
            
            seq_len = len(content_id)

            if seq_len >= self.max_seq:
                content_id_seq = content_id[-self.max_seq:]
                answered_correctly_seq = answered_correctly[-self.max_seq:]
            else:
                content_id_seq[:seq_len] = content_id
                answered_correctly_seq[:seq_len] = answered_correctly

        x = content_id_seq[:].copy()
        x += (answered_correctly_seq[:] == 1) * self.n_skill
        
        questions = np.append(content_id_seq, [target_id])
        x = np.append(x, [0])
        
        return x, questions

##################################

prev_test_df = None
print('start testing')

for (test_df, sample_prediction_df) in tqdm(iter_test):
    
    if (prev_test_df is not None) & (psutil.virtual_memory().percent<90):
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
            
            if len(group[prev_user_id][0]) > params.max_seq:
                new_group_content = group[prev_user_id][0][-params.max_seq:]
                new_group_answered_correctly = group[prev_user_id][1][-params.max_seq:]
                group[prev_user_id] = (new_group_content, new_group_answered_correctly)
                
    prev_test_df = test_df.copy()
    test_df = test_df[test_df.content_type_id == False]
    
    test_dataset = TestDataset(group, test_df, n_skill, max_seq=params.max_seq)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_df), shuffle=False)

    item = next(iter(test_dataloader))
    qa = item[0].to(device).long()
    q = item[1].to(device).long()
    
    with torch.no_grad():
        output= model.predict(q, qa)
        
    output = torch.sigmoid(output)
    output = output[:, -1]
    test_df['answered_correctly'] = output.cpu().numpy()
    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])

print('finish')