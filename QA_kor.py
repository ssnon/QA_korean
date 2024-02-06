# -*- coding: utf-8 -*-
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorWithPadding, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AdamW
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
from torch.utils.data import DataLoader

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

data = pd.read_csv('open/train.csv')

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_special_tokens=True)
max_seq_length = max(max(len(tokenizer.encode(row['질문_1'] + tokenizer.eos_token + row['답변_1'])), len(tokenizer.encode(row['질문_2'] + tokenizer.eos_token + row['답변_1']))) for _, row in data.iterrows())

def preprocess_function(data_):
    formatted_data = []  
    for _, row in tqdm(data_.iterrows()):
        for q_col in ['질문_1', '질문_2']:
            for a_col in ['답변_1', '답변_2', '답변_3', '답변_4', '답변_5']:
                input_text = row[q_col] + tokenizer.eos_token + row[a_col]
                input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=max_seq_length, padding="max_length", truncation=True, return_token_type_ids=False)
                input_ids = input_ids.to(device)
                input_ids = input_ids.squeeze(0)
                formatted_data.append(input_ids)
    return formatted_data
    
def preprocess_function_val(data_):
    formatted_data = []
    for _, row in tqdm(data_.iterrows()):
        for q_col in ['질문_1', '질문_2']:
            for a_col in ['답변_1', '답변_2', '답변_3', '답변_4', '답변_5']:
                input_ids = tokenizer.encode(row[q_col]+ tokenizer.eos_token, return_tensors='pt', max_length=max_seq_length, padding="max_length", truncation=True, return_token_type_ids=False)
                input_ids = input_ids.to(device)
                input_ids = input_ids.squeeze(0)
                #input_ids = torch.tensor(input_ids)
                formatted_data.append([input_ids, row[a_col]])
    return formatted_data
        
data_train, data_val = train_test_split(data, test_size = 0.2, shuffle=True) 
tokenized_data_train = preprocess_function(data_train)
tokenized_data_val = preprocess_function_val(data_val)

train_dataloader = DataLoader(dataset=tokenized_data_train, batch_size=args.batch_size, shuffle=True)
#val_dataloader = DataLoader(dataset=tokenized_data_val,batch_size=1, shuffle=False)   
val_dataloader = DataLoader(dataset=tokenized_data_val,batch_size=args.batch_size, shuffle=False)         
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
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)    
model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=r_, lora_alpha=16, lora_dropout=0.1, bias="all"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
optimizer = AdamW(model.parameters(), lr=lr)   
    
##model evaluate flag
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0

embed_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

def compute_metrics(eval_pred, gt):
    model.eval()
    model.config.use_cache = True
    sample_scores = []
    
    pred_embed = embed_model.encode(eval_pred)
    gt_embed = embed_model.encode(gt)
    
    sample_score = cosine_similarity(gt_embed, pred_embed)
    
    sample_score = max(sample_score, 0)
    print('예측 : ', eval_pred)
    print('정답 : ', gt)
    print('Cosine Similarity Score : ', sample_score)
    print('-'*20)
    sample_scores.append(sample_score)
    print('전체 샘플의 Cosine Similarity Score 평균 : ', np.mean(sample_scores))
    return np.mean(sample_scores)
    
##train process              
for epoch in range(num_epochs):
    total_loss = 0
    progress_bar = tqdm(train_dataloader)
    for batch_idx, batch in enumerate(progress_bar):
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        progress_bar.set_description(f"Epoch {epoch+1} - Avg Loss: {total_loss / (batch_idx+1):.4f}")

    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(tokenized_data_train)}")
    
    ## validation
    if epoch%10 ==9:
        progress_bar = tqdm(val_dataloader)
        total_cossim = 0
        for batch_idx, val in enumerate(progress_bar):
            test_question = val[0]
            gt = val[1]
            output_sequences = model.generate(
                input_ids=test_question.to(device),
                max_length=300,
                temperature=0.9,
                top_k=1,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                num_return_sequences=1

            )
            full_text = tokenizer.decode(output_sequences[0], skip_special_tokens=False)
            answer_start = full_text.find(tokenizer.eos_token) + len(tokenizer.eos_token)
            answer_only = full_text[answer_start:].strip()
            answer_only = answer_only.replace('\n', ' ')
        
            cossim = compute_metrics(answer_only, gt)
            total_cossim += cossim
        

            progress_bar.set_description(f"Epoch {epoch+1}")
            model.train() 
        
        avg_cossim = total_cossim / len(tokenized_data_val)

        print(f"Epoch {epoch+1}/{num_epochs}, Average cossim: {avg_cossim}")

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
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
