from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, AutoTokenizer, HfArgumentParser
from utils.my_trainer import CustomTrainer
from utils.utils import my_compute_metrics,seed_everything
from typing import Optional
from rouge import Rouge
from dataclasses import dataclass, field
from model.my_model import MyCustomModel
from peft import LoraConfig
from datasets import load_dataset, concatenate_datasets
from utils.data_collator import MyDataCollatorForLanguageModelingTest
from utils.eval_utils import calculate_f1score, compute_bleu_scores


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    # system config
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    seed: Optional[int] = field(default=42, metadata={"help": "seed"})

    # model
    llm_name: Optional[str] = field(default="mistralai/Mistral-7B-Instruct-v0.2", metadata={"help": "the model name，  meta-llama/Llama-2-7b-chat-hf "})
    clip_name: Optional[str] = field(default="conch", metadata={"help": "the model name，  conch / pathclip-base / uni"})
    max_seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    
    # data
    dataset_name_list: Optional[str] = field(default="CNX-PathLLM/PVQAClean", metadata={"help": ""})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    
    # eval
    batch_size: Optional[int] = field(default=4, metadata={"help": "batch_size"})
    ckpt_path: Optional[str] = field(default="/bask/homes/a/asiw9691/PathVLM/source/PathLLM/Conch_Mistral_Instruct3/ckpt-800.bin", metadata={"help": "ckpt path"})

device = 'cuda'

def formatting_func(examples):
    question = examples["question"]
    answer = examples["answer"]
    if answer in ["yes","no"]:
        question += "Answer yes or no only!"
    text = f"<|Question|> {question}{tokenizer.eos_token}"
    examples["text"] = text
    return examples

def tokenize(element):
    outputs = tokenizer(
        element,
        add_special_tokens=True,
        truncation=True,
        padding=False,
        max_length=script_args.max_seq_length,
        return_overflowing_tokens=False,
        return_length=False,
    )

    return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}
    
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
seed_everything(script_args.seed)

# set up tokenizer
tokenizer = AutoTokenizer.from_pretrained(script_args.llm_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.truncation_side = 'left'

new_tokens = ['<|Question|>',  '<|Answer|>', '<|Image|>']  # 你要添加的特殊字符列表
num_added_toks = tokenizer.add_tokens(new_tokens)
new_tokens_ids = tokenizer.convert_tokens_to_ids(new_tokens)
print("new_tokens_ids: ", new_tokens_ids)

dataset = []
for dataset_name in script_args.dataset_name_list.split(","):
    dataset.append(load_dataset(dataset_name, split="train", cache_dir="/bask/projects/p/phwq4930-gbm/Zeyu/PathVLM/.cache"))
dataset = concatenate_datasets(dataset)
dataset = dataset.train_test_split(test_size=0.1)
eval_dataset = dataset['test']
eval_dataset = eval_dataset.map(formatting_func, num_proc=4, remove_columns=["question"])

model = MyCustomModel(False, 
                      script_args.clip_name,
                      script_args.load_in_8bit, 
                      script_args.load_in_4bit, 
                      script_args.llm_name,
                      False, 
                      True, 
                      tokenizer,
                      new_tokens_ids[-1])

model.load_state_dict(torch.load(script_args.ckpt_path, map_location=device))
model.to(device)

data_collator = MyDataCollatorForLanguageModelingTest(tokenizer, model.image_processor)

tokenized_dataset = eval_dataset.map(
            tokenize,
            batched=False,
            remove_columns=['text'],
            num_proc=4,
            batch_size=script_args.batch_size,
            input_columns=['text'],
       )

dataloader_params = {
            "batch_size": script_args.batch_size,
            "collate_fn": data_collator,
        }

eval_dataloader = DataLoader(tokenized_dataset, **dataloader_params)

close_ques_acc = 0
close_ques_num = 0
open_ques_f1 = []

open_candidate = []
open_reference = []
close_candidate = []
close_reference = []

for batch in tqdm(eval_dataloader):
    input_ids = batch['input_ids'].to(device)
    attention_masks = batch['attention_mask'].to(device)
    images = batch['image'].to(device)
    labels = batch['labels'].to(device)
    answers = batch['answers']

    # 执行模型推断
    res = model.generate(input_ids=input_ids,
                         attention_mask=attention_masks,
                         labels=labels,
                         image=images)
    for i in range(len(answers)):
        if answers[i] in ['yes','no']:
            close_candidate.append(res[i])
            close_reference.append(answers[i])
        else:
            open_candidate.append(res[i])
            open_reference.append(answers[i])
            
rouge = Rouge()

# calculate accuracy for yes or no problem
for i in range(len(close_reference)):
    close_ques_num += 1
    if close_reference[i] in close_candidate[i]:
        close_ques_acc += 1
        
# calculate f1 score for open ended problem
for i in range(len(open_reference)):
    f1_score = calculate_f1score(open_candidate[i], open_reference[i])
    open_ques_f1.append(f1_score)
    
open_ques_rouge = rouge.get_scores(open_candidate, open_reference, avg=True)
open_bleu_score = compute_bleu_scores(open_candidate, open_reference, avg=True)
open_ques_f1 = np.mean(open_ques_f1)
close_ques_acc = close_ques_acc/close_ques_num

print("open question macro f1_score: {}".format(open_ques_f1))
print("open question rouge score: {}".format(open_ques_rouge))
print("open question bleu score: {}".format(open_bleu_score))
print("close question accuray: {}".format(close_ques_acc))