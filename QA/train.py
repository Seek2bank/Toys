import os
# os.environ["CUDA_VISIBLE_DEVICES"]=1
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from dataset import QADataset, load_data

from torch.utils.data import DataLoader
import random, numpy as np, argparse
from types import SimpleNamespace # TODO 搞清楚这个怎么用
from torch.optim import AdamW

from tqdm import tqdm
import matplotlib.pyplot as plt

from evaluation import model_eval, model_eval_test

os.environ["CUDA_VISIBLE_DEVICES"]="0"

model_dir = "./model"
data_dir = "./data/DuReaderQG"
# tokenizer = T5Tokenizer.from_pretrained(model_dir)


TQDM_DISABLE = False


class Mengzi(nn.Module):
    def __init__(self, config):
        super(Mengzi, self).__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_dir,low_cpu_mem_usage=True)
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir,low_cpu_mem_usage=True)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, context, atten1, query, atten2, answer, atten3):
        # print(answer.shape)
        # print(context.shape)
        if context.device != answer.device:
            print(context.device)
            answer = answer.to(context.device)  
        out = self.t5(context, atten1, labels=answer)
        output = out.logits
        loss = out.loss
        # print(loss)
        
        # last_hidden_states, out = {v for k,v in self.t5(context, atten1, answer, atten3).items()}
        # print(out)
        embedding = self.dropout(output) # [batch_size, max_length, ?]
        # print(embedding.shape)

        return embedding, loss

    def predict_answer(self, context, atten1, query, atten2, answer, atten3):
        embed, loss = self.forward(context, atten1, query, atten2, answer, atten3)
        return embed, loss
        
        
def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }
    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_model(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    data = load_data(os.path.join(data_dir, "train.json"), os.path.join(data_dir, "dev.json"))

    data = QADataset(data, args)
    
    train_dataloader = DataLoader(data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=data.collate_fn)
    # dev_dataloader = DataLoader()

    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
            # 'hidden_size': 768,
            'data_dir': '.'}
    config = SimpleNamespace(**config)
    model = Mengzi(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0
    train_accs, train_losses = [], []
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_context, b_q, b_a, b_m = (batch['context_ids'], batch['context'], batch['question'], batch['answer'], batch['merged'])
            atten_c, atten_q, atten_a, atten_m = (batch['attention_mask'], batch['attention_mask_q'], batch['attention_mask_a'], batch['attention_mask_m'])
            # b_ids = b_ids.to(device)
            b_context = torch.LongTensor(b_context).to(device)
            b_q = torch.LongTensor(b_q).to(device)
            b_a = torch.LongTensor(b_a).to(device)
            b_m = torch.LongTensor(b_m).to(device)

            atten_c= torch.LongTensor(atten_c).to(device)
            atten_q= torch.LongTensor(atten_q).to(device)
            atten_a= torch.LongTensor(atten_a).to(device)
            atten_m= torch.LongTensor(atten_m).to(device)

            optimizer.zero_grad()
            logits, loss = model.predict_answer(b_m, atten_m, b_q, atten_q, b_a, atten_a)
            # loss = F.cross_entropy(logits, b_a, reduction='sum')/args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)
        train_losses.append(train_loss)

        train_acc = model_eval(train_dataloader, model, device)
        train_accs.append(train_acc)
        # dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

        if train_acc > best_dev_acc:
            best_dev_acc = train_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}")
    return train_accs, train_losses

def test_model(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    test_data = load_data(os.path.join(data_dir, "train.json"), os.path.join(data_dir, "dev.json"),"test")
    test_data = QADataset(test_data, args)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=test_data.collate_fn)
    
    saved = torch.load(args.filepath)
    config = saved['model_config']
    
    model = Mengzi(config)
    model.load_state_dict(saved['model'])
    model = model.to(device)
    
    # model.eval()
    print("Evaluate on test set...")
    
    results = []
    results = model_eval_test(test_dataloader, model, device)    
    return results
    


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--use_gpu", action='store_true', default=True)
    parser.add_argument("--dev", action='store_true', default = "./data/DuReaderQG/dev.json")
    parser.add_argument("--filepath", default="./output")
    args = parser.parse_args()
    return args

    
if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.epochs}-{args.lr}-qat5.pt' # Save path.
    epoches = range(args.epochs)
    train_accs, train_losses = train_model(args)
    results = test_model(args)
    
    plt.figure()
    plt.plot(epoches, train_accs)
    plt.xlabel('Epoch')
    plt.ylabel('acc')
    plt.legend()
    picture_dir = "./figures/train_acc.pdf"
    plt.savefig(picture_dir)

    plt.figure()
    plt.plot(epoches, train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()
    picture_dir = "./figures/train_loss.pdf"
    plt.savefig(picture_dir)

