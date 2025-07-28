import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
from torch.utils.data import Dataset



dataset_dir = "./data/DuReaderQG"
model_dir = "./model"

class QADataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir,low_cpu_mem_usage=True)
        self.args = args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        context = [x[0] for x in data]
        context_id = [x[1] for x in data]
        question = [x[2] for x in data]
        answer = [x[3] for x in data]

        merged = [pair for pair in zip(context, question)]

        encoding1 = self.tokenizer(context, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        encoding2 = self.tokenizer(question, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        encoding3 = self.tokenizer(answer, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        encoding4 = self.tokenizer(context, question, return_tensors="pt", padding='max_length', truncation=True, max_length=512)

        context_ids =(encoding1.input_ids)
        question_ids = (encoding2.input_ids)
        answer_ids = (encoding3.input_ids)
        merged_ids = (encoding4.input_ids)

        attention_mask = (encoding1.attention_mask)
        attention_mask_q = (encoding2.attention_mask)
        attention_mask_a = (encoding3.attention_mask)
        attention_mask_m = (encoding4.attention_mask)
        
        return (context_id, context_ids, attention_mask, question_ids, attention_mask_q, answer_ids, attention_mask_a, merged_ids, attention_mask_m)
    
    def collate_fn(self, all_data):
        # need mask? yes, added
        context_id, context, attention_mask, question, attention_mask_q, answer, attention_mask_a, merged, attention_mask_m = self.pad_data(all_data)

        batched_data = {
                'context_ids': context_id,
                'attention_mask': attention_mask,
                'attention_mask_q': attention_mask_q,
                'attention_mask_a': attention_mask_a,
                'attention_mask_m': attention_mask_m,
                'context': context,
                'question': question,
                'answer': answer,
                'merged': merged
        }
        return batched_data

def load_data(train_filename, test_filename, split='train'):
    data = []
    num_labels = {}
    if split == "train":
        with open(train_filename, 'r') as fp:
            for line in fp:
            # for i,record in json.loads(fp):
                record = json.loads(line)
                context = record["context"]
                context_id = record["id"]
                question = record["question"]
                answer = record["answer"]
                data.append((context, context_id, question, answer))
        print(f"Loaded {len(data)} {split} examples from {train_filename}")
    else:
        with open(test_filename, 'r') as fp:
            for line in fp:
            # for i,record in json.loads(fp):
                record = json.loads(line)
                context = record["context"]
                context_id = record["id"] # data in json file is this format
                question = record["question"]
                answer = record["answer"]
                data.append((context, context_id, question, answer))
        print(f"Loaded {len(data)} {split} examples from {test_filename}")
    
    return data

# data = load_data()