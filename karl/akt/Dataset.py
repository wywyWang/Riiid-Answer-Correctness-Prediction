from torch.utils.data import Dataset, DataLoader
import numpy as np

ACCEPTED_USER_CONTENT_SIZE = 1
TAGS_NUM = 188
MAX_TAGS_LEN = 6
####################################
## Min part: 1, Max part: 7
## Min tag: 0, Max tag: 187
## Max tags len: 6
####################################

def pack_concept(content_id_list, q_concepts):
    part_list, tags_list = [], []
    for cid in content_id_list:
        part_list.append(q_concepts[cid]['part'])
        tags_list.append(q_concepts[cid]['tags'])

        cur_q_tags_list_len = len(q_concepts[cid]['tags'])
        if cur_q_tags_list_len < MAX_TAGS_LEN:
            for _ in range(MAX_TAGS_LEN-cur_q_tags_list_len):
                tags_list[-1].append(TAGS_NUM)
        elif cur_q_tags_list_len > MAX_TAGS_LEN:
            assert Exception('ERROR: Tags Length greater than configure, please change the setting')
    
    assert len(part_list) == len(content_id_list)
    assert len(tags_list) == len(content_id_list)
    return part_list, tags_list

class AKTDataset(Dataset):
    def __init__(self, group, n_skill, q_concepts=None, max_seq=100):
        super(AKTDataset, self).__init__()
        self.samples, self.n_skill, self.max_seq = {}, n_skill, max_seq

        self.user_ids = []
        for _, user_id in enumerate(group.index):
            content_id, answered_correctly, explained = group[user_id]
            #######################################################
            # Main Contribution
            if len(content_id) > self.max_seq:
                total_questions = len(content_id)
                initial = total_questions % self.max_seq

                if initial >= ACCEPTED_USER_CONTENT_SIZE:
                    self.user_ids.append(f"{user_id}_0")
                    # part_list, tags_list = pack_concept(content_id[:initial], q_concepts)
                    self.samples[f"{user_id}_0"] = \
                        (content_id[:initial], answered_correctly[:initial], explained[:initial])
                
                for seq in range(total_questions // self.max_seq):
                    self.user_ids.append(f"{user_id}_{seq+1}")
                    start = initial + seq * self.max_seq
                    end = start + self.max_seq
                    # part_list, tags_list = pack_concept(content_id[start:end], q_concepts)
                    self.samples[f"{user_id}_{seq+1}"] = \
                        (content_id[start:end], answered_correctly[start:end], explained[start:end])
            else:
                user_id = str(user_id)
                self.user_ids.append(user_id)
                # part_list, tags_list = pack_concept(content_id, q_concepts)
                self.samples[user_id] = \
                    (content_id, answered_correctly, explained)
                
    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        content_id, answered_correctly, explained = self.samples[user_id]
        seq_len = len(content_id)
        
        content_id_seq = np.zeros(self.max_seq, dtype=int)
        answered_correctly_seq = np.zeros(self.max_seq, dtype=int)
        # part_seq = np.zeros(self.max_seq, dtype=int)
        # tags_seq = np.zeros((self.max_seq, MAX_TAGS_LEN), dtype=int)
        learned_seq = np.zeros(self.max_seq, dtype=int)
        if seq_len >= self.max_seq:
            content_id_seq[:] = content_id[-self.max_seq:]
            answered_correctly_seq[:] = answered_correctly[-self.max_seq:]
            # part_seq[:] = part_list[-self.max_seq:]
            # tags_seq[:] = tags_list[-self.max_seq:]
            learned_seq[:-1] = explained[-self.max_seq+1:]
            padding_mask = []
        else:
            content_id_seq[-seq_len:] = content_id
            answered_correctly_seq[-seq_len:] = answered_correctly
            # part_seq[:seq_len] = part_list
            # tags_seq[:seq_len] = tags_list
            learned_seq[-seq_len:-1] = explained[1:seq_len]
            padding_mask = list(range(self.max_seq - seq_len))
            
        q_data = content_id_seq[:]
        label = answered_correctly_seq[:]
        label[padding_mask] = -1

        qa = content_id_seq[:].copy()
        qa += (answered_correctly_seq[:] == 1) * self.n_skill
        
        # learned_seq += 1
        # qa *= learned_seq

        return qa, q_data, label, learned_seq

class TestDataset(Dataset):
    def __init__(self, samples, test_df, n_skill, max_seq):
        super(TestDataset, self).__init__()
        self.samples = samples
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