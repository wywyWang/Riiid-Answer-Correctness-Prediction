from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

ACCEPTED_USER_CONTENT_SIZE = 1
MAX_TAGS_LEN = 6
TAGS_NUM = 188


class SAKTDataset(Dataset):
    def __init__(self, group, questions, n_skill, max_seq=100):
        super(SAKTDataset, self).__init__()
        self.samples, self.n_skill, self.max_seq = {}, n_skill, max_seq
        
        self.user_ids = []
        for i, user_id in enumerate(tqdm(group.index)):
            content_id, answered_correctly = group[user_id]
            tags = [questions[idx] for idx in content_id]

            # Main Contribution
            if len(content_id) > self.max_seq:
                total_questions = len(content_id)
                initial = total_questions % self.max_seq

                if initial >= ACCEPTED_USER_CONTENT_SIZE:
                    self.user_ids.append(f"{user_id}_0")
                    self.samples[f"{user_id}_0"] = (content_id[:initial], answered_correctly[:initial], tags[:initial])
                
                for seq in range(total_questions // self.max_seq):
                    self.user_ids.append(f"{user_id}_{seq+1}")
                    start = initial + seq * self.max_seq
                    end = start + self.max_seq
                    self.samples[f"{user_id}_{seq+1}"] = (content_id[start:end], answered_correctly[start:end], tags[start:end])
            else:
                user_id = str(user_id)
                self.user_ids.append(user_id)
                self.samples[user_id] = (content_id, answered_correctly, tags)
                
    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        content_id, answered_correctly, tags = self.samples[user_id]
        seq_len = len(content_id)
        
        content_id_seq = np.zeros(self.max_seq, dtype=int)
        answered_correctly_seq = np.zeros(self.max_seq, dtype=int)
        tags_seq = np.full((self.max_seq, MAX_TAGS_LEN), fill_value=TAGS_NUM, dtype=int)

        if seq_len >= self.max_seq:
            content_id_seq[:] = content_id[-self.max_seq:]
            answered_correctly_seq[:] = answered_correctly[-self.max_seq:]
            tags_seq[:] = tags[-self.max_seq:]
        else:
            content_id_seq[-seq_len:] = content_id
            answered_correctly_seq[-seq_len:] = answered_correctly
            tags_seq[-seq_len:] = tags
            
        target_id = content_id_seq[1:]
        label = answered_correctly_seq[1:]
        tags_id = tags_seq[1:]
        
        x = content_id_seq[:-1].copy()
        x += (answered_correctly_seq[:-1] == 1) * self.n_skill
        
        return x, target_id, label, tags_id

class TestDataset(Dataset):
    def __init__(self, samples, test_df, question_df, n_skill, max_seq=100):
        super(TestDataset, self).__init__()
        self.samples = samples
        self.user_ids = [x for x in test_df["user_id"].unique()]
        self.test_df = test_df
        self.question_df = question_df
        self.n_skill = n_skill
        self.max_seq = max_seq

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