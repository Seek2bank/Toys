from train import Mengzi
import os

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

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

base_dir = "./model"
model_dir = "./output/5-1e-05-qat5.pt"
data_dir = "./data/DuReaderQG"
tokenizer = T5Tokenizer.from_pretrained(base_dir)

config = {'hidden_dropout_prob': 0.3,
            # 'hidden_size': 768,
            'data_dir': '.'}

saved = torch.load(model_dir)
config = saved['model_config']
    
model = Mengzi(config)
model.load_state_dict(saved['model'])
model = model.to("cpu")
optimizer = torch.optim.AdamW(model.parameters())

context = """年基准利率4.35%。 从实际看,贷款的基本条件是: 一是中国大陆居民,年龄在60岁以下; 二是有稳定的住址和工作或经营地点; 三是有稳定的收入来源; 四是无不良信用记录,贷款用途不能作为炒股,赌博等行为; 五是具有完全民事行为能力。"""
question = "2017年银行贷款基准利率"
input = tokenizer(context, question, return_tensors="pt", padding = "max_length", max_length=512).input_ids
atten1 = tokenizer(context, question, return_tensors="pt", padding = "max_length", max_length=512).attention_mask

input=input.to("cpu")
atten1 = atten1.to("cpu")

output = model.t5.generate(input).cpu().numpy()
output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
print("text")
print(output_text)


