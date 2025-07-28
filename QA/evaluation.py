import torch
from torchtext.data.metrics import bleu_score
from sacrebleu.metrics import BLEU
from tqdm import tqdm
import numpy as np
import json
import jieba

bleu = BLEU(tokenize='zh')

TQDM_DISABLE = False

def model_eval(dataloader, model, device):
    model.eval()
    preds = []
    labels = []

    
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval')):
        with torch.no_grad():
            b_ids, b_context, b_q, b_a, b_m = (batch['context_ids'], batch['context'], batch['question'], batch['answer'], batch['merged'])
            atten_c, atten_q, atten_a, atten_m = (batch['attention_mask'], batch['attention_mask_q'], batch['attention_mask_a'], batch['attention_mask_m'])
            
            b_m = b_m.to(device)
            
            generate_tokens = model.t5.generate(b_m).cpu().numpy()
            label_tokens = b_a.cpu().numpy()

            decoded_preds = model.tokenizer.batch_decode(generate_tokens, skip_special_tokens=True)
            decoded_labels = model.tokenizer.batch_decode(label_tokens, skip_special_tokens=True)
            
            preds += [pred.strip() for pred in decoded_preds]
            labels += [[label.strip()] for label in decoded_labels]
            # preds += [list(jieba.cut(pred)) for pred in decoded_preds]
            # labels += [[list(jieba.cut(label))] for label in decoded_labels]
            # preds.extend([list(jieba.cut(p)) if p.strip() else [""] for p in decoded_preds])
            # labels.extend([[list(jieba.cut(l)) if l.strip() else [""]] for l in decoded_labels])
            
    bleu = BLEU(tokenize='zh')
    # for p, l in zip(preds, labels):
    #     print("Pred:", p)
    #     print("Label:", l)
    # print(preds)
    # print(labels)
    return bleu.corpus_score(preds, labels).score
    
def model_eval_test(dataloader, model, device):
    model.eval()
    preds = []
    labels = []
    sources = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc=f'eval')):
            b_ids, b_context, b_q, b_a, b_m = (batch['context_ids'], batch['context'], batch['question'], batch['answer'], batch['merged'])
            atten_c, atten_q, atten_a, atten_m = (batch['attention_mask'], batch['attention_mask_q'], batch['attention_mask_a'], batch['attention_mask_m'])
            
            b_m = b_m.to(device)
            generate_tokens = model.t5.generate(b_m).cpu().numpy()
            label_tokens = b_a.cpu().numpy()

            decoded_preds = model.tokenizer.batch_decode(generate_tokens, skip_special_tokens=True)
            decoded_labels = model.tokenizer.batch_decode(label_tokens, skip_special_tokens=True)
            decoded_sources = model.tokenizer.batch_decode(b_m, skip_special_tokens=True)
            
            preds += [pred.strip() for pred in decoded_preds]
            labels += [[label.strip()] for label in decoded_labels]
            sources += [source.strip() for source in decoded_sources]
        
        bleu = BLEU(tokenize='zh')
        bleu_score =  bleu.corpus_score(preds, labels).score
        print(f"Test BLEU: {bleu_score:>0.2f}\n")
        results = []
        print("saving predicted results...")
        for source, label, pred in zip(sources, preds, labels):
            results.append({
                "sentence": source,
                "prediction": pred,
                "label": label
            })
        with open('test_data_pred.json', 'wt', encoding='utf-8') as f:
            for exapmle_result in results:
                f.write(json.dumps(exapmle_result, ensure_ascii=False) + '\n')
        return results
    
