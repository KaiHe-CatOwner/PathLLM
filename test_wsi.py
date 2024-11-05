from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import re
import torch
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, HfArgumentParser
from utils.utils import seed_everything
from typing import Optional
from dataclasses import dataclass, field
from utils.data_collator import MyDataCollatorForWPathVLM
from datasets import load_dataset, concatenate_datasets, load_from_disk
from model.my_model import WPathVLM
from peft import LoraConfig, get_peft_model

from utils.eval_utils import calculate_prf_score, compute_bleu_scores, split_sentence, compute_cider_scores, compute_spice_scores

device = 'cuda'

questions = pd.read_csv('./utils/question_wsi_list.csv', header=None)  
questions = questions[0].tolist()

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine-tune with SFTTrainer.
    """
    # System config
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "Load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "Load the model in 4 bits precision"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    seed: Optional[int] = field(default=42, metadata={"help": "Random seed"})
    
    # Model
    llm_name: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B", metadata={"help": "The model name"})
    max_seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    llm_requires_grad: Optional[bool] = field(default=False, metadata={"help": "True or  /output/checkpoint-1400"})
    
    # Data
    select_data_num: Optional[int] = field(default=-1, metadata={"help": "the number of training data， -1 mean use all data"})
    dataset_name_list: Optional[str] = field(default="CNX-PathLLM/TCGA-WSI-Text", metadata={"help": "Dataset names separated by comma"})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "The text field of the dataset"})
    data_local_dir: Optional[str] = field(default=None, metadata={"help": "Local directory to load data from"})
    eval_fold_index: Optional[int] = field(default=9, metadata={"help": "The test fold index"})
    data_cache_dir: Optional[str] = field(default="~/.cache", metadata={"help": "Cache directory for dataset and model"})
    results_save_path: Optional[str] = field(default="output.csv", metadata={"help": "the save path for prediction results"})
    fea_root: Optional[str] = field(default="/bask/homes/a/asiw9691/PathVLM/WSI_Dataset/Conch/", metadata={"help": "the root path for WSI feature"})
    
    # WSI hyperparameters
    n_heads: Optional[str] = field(default='32,16,8', metadata={"help": "Number of attention heads for WSI aggregation"})
    n_level: Optional[int] = field(default=3, metadata={"help": "Number of hierarchical levels for WSI embedding"})
    embed_dim: Optional[int] = field(default=512, metadata={"help": "Embedding dimension of each patch"})
    agg_strategy: Optional[str] = field(default='abmil', metadata={"help": "the strategy for WSI aggregation, sample, kmeans, gmm, abmil"})
    
    # Evaluation
    batch_size: Optional[int] = field(default=4, metadata={"help": "Batch size"})
    ckpt_path: Optional[str] = field(default=None, metadata={"help": "Checkpoint path"})
    shuffle: Optional[bool] = field(default=False, metadata={"help": "shuffle eval_dataloader or not"})
    eval_sample_size: Optional[int] = field(default=-1, metadata={"help": "-1 indicate evaluating on all"})

    #lora
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters, 4 to 64"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters, 16 to 128"})

def formatting_func_des(examples):
    question = random.choice(questions)
    answer = examples["description"]
    text = f"<|Question|>{question} " + f"<|Answer|>"
    examples["text"] = text
    examples["answer"] = answer
    examples["question"] = question
    return examples

def formatting_func_qa_open(examples):
    question = examples["question"]
    answer = examples["answer"]
    text = f"<|Question|>{question}" + f"<|Answer|>"
    examples["text"] = text
    examples["answer"] = answer
    return examples

def formatting_func_qa_close(examples):
    question = examples["question"]
    answer = examples["answer"]
    if answer.lower() in ['yes', 'no']:
        prompt = f" Please provide only the answer (either Yes or No) for the following statement. Do not include any explanations or additional text. Just give Yes or No."
    else:
        prompt = f" Please provide only the answer (for example, A. [Answer Text], B. [Answer Text], etc.) for the following question. Do not include any explanations or additional text. Just give the letter followed by the corresponding answer."
    # text = f"<|Question|>{question}<|Prompt|>{prompt}" + f"<|Answer|>"
    text = f"<|Question|>{question}" + f"<|Answer|>"
    examples["text"] = text
    examples["answer"] = answer
    return examples

def tokenize(element):
    """Tokenize the input text."""
    outputs = tokenizer(
        element,
        add_special_tokens=True,
        truncation=True,
        padding=False,
        max_length=script_args.max_seq_length,
        return_overflowing_tokens=False,
        return_length=False
    )
    return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

def evaluate_model(model, eval_dataloader, script_args, mode='open'):
    """
    Evaluate the model on the provided data loader and save results.
    
    Parameters:
        model: The model to evaluate.
        eval_dataloader: DataLoader for evaluation data.
        device: The device (CPU or GPU) on which to perform computations.
        script_args: Arguments containing settings for evaluation, e.g., eval_sample_size and results_save_path.
    """
    qes_list, ans_list, res_list = [], [], []
    
    total_batches = len(eval_dataloader)
    N = script_args.eval_sample_size if script_args.eval_sample_size != -1 else total_batches

    for i, batch in enumerate(tqdm(eval_dataloader, total=N, desc="Evaluating")):
        if i >= N:
            break

        # Move inputs to the device
        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_mask'].to(device)
        fea0, fea1, fea2 = batch['fea0'].to(device), batch['fea1'].to(device), batch['fea2'].to(device)
        cor0, cor1, cor2 = batch['cor0'].to(device), batch['cor1'].to(device), batch['cor2'].to(device)
        mask0, mask1, mask2 = batch['mask0'].to(device), batch['mask1'].to(device), batch['mask2'].to(device)
        
        questions = batch['questions']
        answers = batch['answers']

        # Model inference
        res = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            fea0=fea0, fea1=fea1, fea2=fea2,
            mask0=mask0, mask1=mask1, mask2=mask2,
            cor0=cor0, cor1=cor1, cor2=cor2,
        )

        # Collect results
        qes_list.extend(questions)
        ans_list.extend(answers)
        res_list.extend(res)

    # Save results in a dictionary
    results = {
        "questions": qes_list,
        "answers": ans_list,
        "results": res_list
    }

    # Evaluate using the specified metrics function
    if mode == 'open':
        metrics_open_ended(res_list, ans_list, script_args)
    else:
        metrics_close_ended(res_list, ans_list, script_args)

    # Save results to a CSV file
    df_results = pd.DataFrame(results)

    filename, ext = os.path.splitext(script_args.results_save_path)

    if mode == 'open':
        save_path = f"{filename}_open{ext}"
    else:
        save_path = f"{filename}_close{ext}"

    df_results.to_csv(save_path, index=False)

    print(f"Results saved to {save_path}")

def metrics_open_ended(open_candidate, open_reference, script_args):
    open_ques_pre = []
    open_ques_rec = []
    open_ques_f1 = []
    open_bleu_score = []


    # calculate f1 score for open ended problem
    for i in range(len(open_reference)):
        precision, recall, f1_score = calculate_prf_score(open_candidate[i], open_reference[i])
        open_ques_pre.append(precision)
        open_ques_rec.append(recall)
        open_ques_f1.append(f1_score)
        
    open_bleu_score = compute_bleu_scores(open_candidate, open_reference, avg=True)
    open_cider_score = compute_cider_scores(open_candidate, open_reference)
    # open_spice_score = compute_spice_scores(open_candidate, open_reference)

    open_ques_pre = np.mean(open_ques_pre)
    open_ques_rec = np.mean(open_ques_rec)
    open_ques_f1 = np.mean(open_ques_f1)

    print("open question macro f1_score: {}\n".format(open_ques_f1))
    print("open question macro precision: {}\n".format(open_ques_pre))
    print("open question macro recall: {}\n".format(open_ques_rec))
    print("open question bleu score: {}\n".format(open_bleu_score))
    print("open question cider score: {}\n".format(open_cider_score))
    # print("open question spice score: {}\n".format(open_spice_score))

    output_path = script_args.ckpt_path.replace('.bin', '.txt')

    with open(output_path, 'a') as file:
        file.write("open question macro f1_score: {}\n".format(open_ques_f1))
        file.write("open question macro precision: {}\n".format(open_ques_pre))
        file.write("open question macro recall: {}\n".format(open_ques_rec))
        file.write("open question bleu score: {}\n".format(open_bleu_score))
        file.write("open question cider score: {}\n".format(open_cider_score))
        # file.write("open question spice score: {}\n".format(open_spice_score))

    print("Results have been written to {}".format(output_path))

def metrics_close_ended(close_candidate, close_reference, script_args):
    correct_predictions = 0
    total_predictions = len(close_candidate)

    for answer, result in zip(close_candidate, close_reference):
        answer = answer.strip().lower()
        result = str(result).strip().lower()
        
        # 判断题的匹配逻辑（“yes”或“no”）
        if answer in ["yes", "no"]:
            if answer in result:
                correct_predictions += 1
        else:
            answer_choice = re.search(r'\b([a-d])\.\s?', answer)
            if answer_choice:
                answer_choice = answer_choice.group(1)
                result_choice = re.search(r'\b([a-d])\.\s?', result)
                if result_choice:
                    result_choice = result_choice.group(1)
                else:
                    result_choice = re.search(r'\b([1-4])\.\s?', result)
                    choice_map = {'1': 'a', '2': 'b', '3': 'c', '4': 'd'}
                    if result_choice:
                        result_choice = choice_map[result_choice.group(1)]
                
                if answer_choice == result_choice:
                    correct_predictions += 1

    print("close question accuracy: {}\n".format(correct_predictions/total_predictions))
    
    output_path = script_args.ckpt_path.replace('.bin', '.txt')

    with open(output_path, 'a') as file:
        file.write("close ended accuracy: {}\n".format(correct_predictions/total_predictions))

    print("Results have been written to {}".format(output_path))
        
# Parse arguments and set seed
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
seed_everything(script_args.seed)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(script_args.llm_name)
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token = "<|finetune_right_pad_id|>"
tokenizer.padding_side = 'left'
tokenizer.truncation_side = 'left'

print(tokenizer.eos_token)

# Add new tokens
new_tokens = ['<|Question|>', '<|Prompt|>', '<|Answer|>', '<|Image|>']
num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
new_tokens_ids = tokenizer.convert_tokens_to_ids(new_tokens)
print("New tokens IDs:", new_tokens_ids)

# Determine data split
split_text = f"test[:{script_args.select_data_num}]" if script_args.select_data_num > 0 else "test"

# if script_args.data_local_dir is None:
open_dataset = []
close_dataset = []

for dataset_name in script_args.dataset_name_list.split(","):
    columns_to_remove = ['slide_id']
    one_dataset = load_dataset(dataset_name, split=split_text, cache_dir=script_args.data_cache_dir)
    if 'project' in one_dataset.column_names:
        columns_to_remove.append('project')
    elif 'site' in one_dataset.column_names:
        columns_to_remove.append('site')

    if 'QA' in dataset_name:  # for QA instruction dataset
        # columns_to_remove += ['question']
        if 'Open' in dataset_name: # for OpenQA instruction dataset
            one_dataset = one_dataset.map(formatting_func_qa_open, num_proc=20, remove_columns=columns_to_remove)
            open_dataset.append(one_dataset)
        else: # for CloseQA instruction dataset
            one_dataset = one_dataset.map(formatting_func_qa_close, num_proc=20, remove_columns=columns_to_remove)
            close_dataset.append(one_dataset)
    else:
        columns_to_remove += ['description']
        one_dataset = one_dataset.map(formatting_func_des, num_proc=20, remove_columns=columns_to_remove)
        open_dataset.append(one_dataset)

if open_dataset!=[]:
    open_dataset = concatenate_datasets(open_dataset)

if close_dataset!=[]:
    close_dataset = concatenate_datasets(close_dataset)

# Load model
print(open_dataset)
print(close_dataset)

model = WPathVLM(script_args.llm_requires_grad, 
                script_args.load_in_8bit, 
                script_args.load_in_4bit, 
                script_args.llm_name, 
                script_args.trust_remote_code, # False
                script_args.token, # True
                tokenizer,
                new_tokens_ids[-1],
                n_heads = script_args.n_heads, 
                n_level = script_args.n_level, 
                embed_dim = script_args.embed_dim,
                agg_strategy = script_args.agg_strategy,
                data_cache_dir = script_args.data_cache_dir,
                )

if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,  # Use a moderate rank
        lora_alpha=script_args.peft_lora_alpha,  # Scaling factor
        bias="none",  # No bias adaptation
        task_type="CAUSAL_LM",  # For causal language modeling tasks
    )
    model.llm = get_peft_model(model.llm, peft_config)
    model.llm.print_trainable_parameters()
else:
    peft_config = None

model.load_state_dict(torch.load(script_args.ckpt_path, map_location=device))
model.to(device)

# Prepare data loader
data_collator = MyDataCollatorForWPathVLM(tokenizer=tokenizer, 
                                        fea_root=script_args.fea_root, 
                                        fea_dim=script_args.embed_dim, 
                                        n_level=script_args.n_level,
                                        n_heads=list(map(int, script_args.n_heads.split(','))),
                                        agg_strategy=script_args.agg_strategy,
                                        test=True)

# if script_args.adaptor == 'qformer':
#         remove_columns = []
#     else:

dataloader_params = {"batch_size": script_args.batch_size, "collate_fn": data_collator, "shuffle": script_args.shuffle}
remove_columns=['text']
if open_dataset!=[]:
    tokenized_open_dataset = open_dataset.map(
                        tokenize,
                        batched=False,
                        remove_columns=remove_columns,
                        num_proc=4,
                        batch_size=script_args.batch_size,
                        input_columns=['text'],
                        )
    open_dataloader = DataLoader(tokenized_open_dataset, **dataloader_params)
    print("### Start evaluating open-ended!")
    evaluate_model(model, open_dataloader, script_args, mode='open')

if close_dataset!=[]:
    tokenized_close_dataset = close_dataset.map(
                        tokenize,
                        batched=False,
                        remove_columns=remove_columns,
                        num_proc=4,
                        batch_size=script_args.batch_size,
                        input_columns=['text'],
                        )
    close_dataloader = DataLoader(tokenized_close_dataset, **dataloader_params)
    print("### Start evaluating close-ended!")
    evaluate_model(model, close_dataloader, script_args, mode='close')