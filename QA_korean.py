# -*- coding: utf-8 -*-
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorWithPadding
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import evaluate
import torch
import numpy as np
from peft import AutoPeftModelForCausalLM, PeftModelForTokenClassification
import os
from datasets import concatenate_datasets,DatasetDict
import pandas as pd
from sentence_transformers import SentenceTransformer # SentenceTransformer Version 2.2.2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse, sys


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="EleutherAI/polyglot-ko-12.8b",
                      help=' : load model name')
parser.add_argument('--epoch', type=int, default=20,
                      help=' : epochs')
parser.add_argument('--batch_size', type=int, default=4,
                      help=' : batch size')
parser.add_argument('--rank', type=int, default=12,
                      help=' : lora rank')
parser.add_argument('--lr', type=int, default=1e-4,
                      help=' : learning rate')

args = parser.parse_args()


os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_checkpoint = args.model
lr = args.lr
batch_size = args.batch_size
num_epochs = args.epoch
r_=args.rank


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not os.path.exists('experiment'):
    os.mkdir('experiment')
output_dir =  f'experiment/{model_checkpoint}_{r_}'

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
data = pd.read_csv('open/train.csv')
def preprocess_function(data):
    formatted_data = []
    for _, row in tqdm(data.iterrows()):
        for q_col in ['질문_1', '질문_2']:
            for a_col in ['답변_1', '답변_2', '답변_3', '답변_4', '답변_5']:
                # 질문과 답변 쌍을 </s> token으로 연결
                input_text = row[q_col] + tokenizer.eos_token + row[a_col]
                input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True)
                formatted_data.append(input_ids)
    return formatted_data
    
data_train, data_val = train_test_split(data, test_size = 0.2, shuffle=True) 
tokenized_data_train = preprocess_function(data_train)
tokenized_data_val = preprocess_function(data_val)

'''            
for _, row in tqdm(data.iterrows()):
    for q_col in ['질문_1', '질문_2', 'category']:
        for a_col in ['답변_1', '답변_2', '답변_3', '답변_4', '답변_5']:
            # 질문과 답변 쌍을 </s> token으로 연결
            input_text = row[q_col] + tokenizer.eos_token + row[a_col]
            #input_ids = tokenizer.encode(input_text, return_tensors='pt')
            formatted_data.append(input_text)
'''
              
#data = load_dataset("beomi/KoAlpaca-v1.1a")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)    
model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)
peft_config = LoraConfig(
    task_type=TaskType.QUESTION_ANS, inference_mode=False, r=r_, lora_alpha=16, lora_dropout=0.1, bias="all"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
    
    
##model evaluate flag
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0

embed_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

def compute_metrics(eval_pred):
    pred, gt = eval_pred
    model.eval()
    model.config.use_cache = True
    sample_scores = []
    
    pred_embed = embed_model.encode(pred)
    gt_embed = embed_model.encode(gt)
    
    sample_score = cosine_similarity(gt_embed, pred_embed)
    
    sample_score = max(sample_score, 0)
    print('예측 : ', pred)
    print('정답 : ', gt)
    print('Cosine Similarity Score : ', sample_score)
    print('-'*20)
    sample_scores.append(sample_score)
    print('전체 샘플의 Cosine Similarity Score 평균 : ', np.mean(sample_scores))
    return np.mean(sample_scores)
        
          
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=5,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data_train,
    eval_dataset=tokenized_data_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[MyCallback(model)],
)

trainer.train()

output_dir_1 = os.path.join(output_dir, "result_epoch{num_epochs}.pth")
torch.save(model.state_dict(), output_dir_1)

'''
data_test = pd.read_csv('open/test.csv')
tokenized_data_test = data_test.map(preprocess_function)


model.eval()
model.config.use_cache = True
sample_scores = []
for pred, gt in zip(preds, gts):
    # 생성된 답변 내용을 512 Embedding Vector로 변환
    pred_embed = embed_model.encode(pred)
    gt_embed = embed_model.encode(gt)
    
    sample_score = cosine_similarity(gt_embed, pred_embed)
    # Cosine Similarity Score가 0보다 작으면 0으로 간주
    sample_score = max(sample_score, 0)
    print('예측 : ', pred)
    print('정답 : ', gt)
    print('Cosine Similarity Score : ', sample_score)
    print('-'*20)
    sample_scores.append(sample_score)
print('전체 샘플의 Cosine Similarity Score 평균 : ', np.mean(sample_scores))
'''
  
trainer.push_to_hub()

REPO_NAME = output_dir # ex) 'my-bert-fine-tuned'
AUTH_TOKEN = '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' # <https://huggingface.co/settings/token>

model.push_to_hub(
    REPO_NAME, 
    use_temp_dir=True, 
    use_auth_token=AUTH_TOKEN
)
tokenizer.push_to_hub(
    REPO_NAME, 
    use_temp_dir=True, 
    use_auth_token=AUTH_TOKEN
)
