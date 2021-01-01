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

from torch.utils.data import Dataset, DataLoader
from run import train, test
# from utils import try_makedirs, load_model, get_file_name_identifier
from akt import AKT

import Dataset as DS

from global_var import *

def try_makedirs(path_):
    if not os.path.isdir(path_):
        try:
            os.makedirs(path_)
        except FileExistsError:
            pass

def train_one_dataset(params, train_dataloader, valid_dataloader):
    # ================================== model initialization ==================================
    model = AKT(n_question=params.n_question, n_blocks=params.n_block, d_model=params.d_model,
                    dropout=params.dropout, kq_same=params.kq_same, l2=params.l2, n_heads=params.n_head,
                    d_ff=params.d_ff, final_fc_dim=params.final_fc_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.999), eps=1e-8)
    print("\n")
    # ================================== start training ==================================
    all_train_loss = {}
    all_train_accuracy = {}
    all_train_auc = {}
    all_valid_loss = {}
    all_valid_accuracy = {}
    all_valid_auc = {}
    best_valid_auc = 0

    for idx in range(params.max_iter):
        # Train Model
        train_loss, train_accuracy, train_auc = train(model, params, optimizer, train_dataloader)
        
        print('epoch', idx + 1)
        print("train_auc: ", train_auc)
        print("train_accuracy: ", train_accuracy)
        print("train_loss: ", train_loss)

        # Validation step
        valid_loss, valid_accuracy, valid_auc = test(model, params, optimizer, valid_dataloader)

        print("valid_auc: ", valid_auc)
        print("valid_accuracy: ", valid_accuracy)
        print("valid_loss: ", valid_loss)
        print('================')

        try_makedirs('model')

        all_valid_auc[idx + 1] = valid_auc
        all_train_auc[idx + 1] = train_auc
        all_valid_loss[idx + 1] = valid_loss
        all_train_loss[idx + 1] = train_loss
        all_valid_accuracy[idx + 1] = valid_accuracy
        all_train_accuracy[idx + 1] = train_accuracy

        # output the epoch with the best validation auc
        if valid_auc > best_valid_auc:
            path = os.path.join('model', params.file_name) + '_*'
            for i in glob.glob(path):
                os.remove(i)
            best_valid_auc = valid_auc
            best_epoch = idx+1
            torch.save({'epoch': idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                        },
                       os.path.join('model', params.file_name)+'_' + str(idx+1)
                       )
        if idx-best_epoch > 40:
            break   

    try_makedirs('result')
    f_save_log = open(os.path.join('result', params.file_name), 'w')
    f_save_log.write("valid_auc:\n" + str(all_valid_auc) + "\n\n")
    f_save_log.write("train_auc:\n" + str(all_train_auc) + "\n\n")
    f_save_log.write("valid_loss:\n" + str(all_valid_loss) + "\n\n")
    f_save_log.write("train_loss:\n" + str(all_train_loss) + "\n\n")
    f_save_log.write("valid_accuracy:\n" + str(all_valid_accuracy) + "\n\n")
    f_save_log.write("train_accuracy:\n" + str(all_train_accuracy) + "\n\n")
    f_save_log.close()
    return best_epoch

def load_questions_part_and_tags(que_pth, que_num):
    res = []
    with open(que_pth) as f:
        next(f)
        for row in f.readlines():
            data = row.strip().split(',')
            element = dict()
            part, tags = int(data[-2]), data[-1].split(' ')
            element['part'] = part
            element['tags'] = []
            for tag in tags:
                if tag == '':
                    continue
                element['tags'].append(int(tag))
            res.append(element)

    assert len(res) == que_num
    return res

if __name__ == '__main__':
    # Parse Arguments
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

    # DKVMN Specific  Parameter
    # parser.add_argument('--q_embed_dim', type=int, default=50,
    #                     help='question embedding dimensions')
    # parser.add_argument('--qa_embed_dim', type=int, default=256,
    #                     help='answer and question embedding dimensions')
    # parser.add_argument('--memory_size', type=int,
    #                     default=50, help='memory size')
    # parser.add_argument('--init_std', type=float, default=0.1,
    #                     help='weight initialization std')
    # DKT Specific Parameter
    # parser.add_argument('--hidden_dim', type=int, default=512)
    # parser.add_argument('--lamda_r', type=float, default=0.1)
    # parser.add_argument('--lamda_w1', type=float, default=0.1)
    # parser.add_argument('--lamda_w2', type=float, default=0.1)
    parser.add_argument('--max_seq', type=int, default=200)
    parser.add_argument('--file_name', type=str, default='output')

    # # Datasets and Model
    # parser.add_argument('--model', type=str, default='akt_pid',
    #                     help="combination of akt/sakt/dkvmn/dkt (mandatory), pid/cid (mandatory) separated by underscore '_'. For example tf_pid")

    params = parser.parse_args()

    TRAIN_SAMPLES = 320000
    #######################################
    ## Load Data
    # dtypes = {'timestamp': 'int64', 'user_id': 'int32' ,'content_id': 'int16','content_type_id': 'int8','answered_correctly':'int8'}
    # train_df = dt.fread('./data/train.csv', columns=set(dtypes.keys())).to_pandas()
    # for col, dtype in dtypes.items():
    #     train_df[col] = train_df[col].astype(dtype)
    # train_df = train_df[train_df.content_type_id == False]
    # train_df = train_df.sort_values(['timestamp'], ascending=True)
    # train_df.reset_index(drop=True, inplace=True)

    ## Preprocess
    # skills = train_df["content_id"].unique()
    # joblib.dump(skills, "skills.pkl.zip")
    skills = joblib.load('skills.pkl.zip')
    n_skill = len(skills)
    params.n_question = n_skill
    print("number skills", len(skills))

    # group = train_df[['user_id', 'content_id', 'answered_correctly']].groupby('user_id').apply(lambda r: (
    #             r['content_id'].values,
    #             r['answered_correctly'].values))
    # joblib.dump(group, "group.pkl.zip")
    group = joblib.load('group.pkl.zip')
    # del train_df
    gc.collect()

    print('preparing indexes and group')
    train_indexes = list(group.index)[:TRAIN_SAMPLES]
    valid_indexes = list(group.index)[TRAIN_SAMPLES:]
    train_group = group[group.index.isin(train_indexes)]
    valid_group = group[group.index.isin(valid_indexes)]
    del group, train_indexes, valid_indexes
    print(len(train_group), len(valid_group))

    print('preparing questions concepts')
    q_concepts = load_questions_part_and_tags('./data/questions.csv', params.n_question)

    print('preparing training dataloader')
    train_dataset = DS.AKTDataset(train_group, n_skill, q_concepts, max_seq=params.max_seq)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=12)
    del train_group

    print('preparing validation dataloader')
    valid_dataset = DS.AKTDataset(valid_group, n_skill, q_concepts, max_seq=params.max_seq)
    valid_dataloader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False, num_workers=12)
    del valid_group
    ###############################

    np.random.seed(params.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    # file_name_identifier = get_file_name_identifier(params)

    ###Train- Test
    d = vars(params)
    for key in d:
        print('\t', key, '\t', d[key])

    # Train and get the best episode
    train_one_dataset(params, train_dataloader, valid_dataloader)
