from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

def formatting_func_des(examples):
    question = random.choice(questions)
    answer = examples["description"]
    text = f"<Question> {question}{tokenizer.eos_token} " # + f"<Answer> {answer}{tokenizer.eos_token}\n"
    examples["text"] = text
    examples["answer"] = answer
    return examples

def formatting_func_qa(examples):
    question = examples["question"]
    answer = examples["answer"]
    text = f"<Question> {question}{tokenizer.eos_token} " # + f"<Answer> {answer}{tokenizer.eos_token}\n"
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

def metrics_open_ended(open_candidate, open_reference, script_args):
    # todo CIDEr and SPICE
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

# Parse arguments and set seed
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
seed_everything(script_args.seed)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(script_args.llm_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.truncation_side = 'left'

# Add new tokens
new_tokens = ['<Question>',  '<Answer>', '<Image>']
num_added_toks = tokenizer.add_tokens(new_tokens)
new_tokens_ids = tokenizer.convert_tokens_to_ids(new_tokens)
print("New tokens IDs:", new_tokens_ids)

# Determine data split
split_text = f"test[:{script_args.select_data_num}]" if script_args.select_data_num > 0 else "test"


# if script_args.data_local_dir is None:
dataset = []

for dataset_name in script_args.dataset_name_list.split(","):
    columns_to_remove = ['slide_id']
    one_dataset = load_dataset(dataset_name, split=split_text, cache_dir=script_args.data_cache_dir)
    if 'project' in one_dataset.column_names:
        columns_to_remove.append('project')
    elif 'site' in one_dataset.column_names:
        columns_to_remove.append('site')

    if 'QA' in dataset_name:
        columns_to_remove += ['question']
        one_dataset = one_dataset.map(formatting_func_qa, num_proc=20, remove_columns=columns_to_remove)
    else:
        columns_to_remove += ['description']
        one_dataset = one_dataset.map(formatting_func_des, num_proc=20, remove_columns=columns_to_remove)
    dataset.append(one_dataset)
    
dataset = concatenate_datasets(dataset)
eval_dataset = dataset

# Load model
print(eval_dataset)

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
remove_columns=['text']
tokenized_dataset = dataset.map(
                tokenize,
                batched=False,
                remove_columns=remove_columns,
                num_proc=4,
                batch_size=script_args.batch_size,
                input_columns=['text'],
        )
            
dataloader_params = {"batch_size": script_args.batch_size, "collate_fn": data_collator, "shuffle": script_args.shuffle}

eval_dataloader = DataLoader(tokenized_dataset, **dataloader_params)

# Evaluate the model
ans_list, res_list = [], []

total_batches = len(eval_dataloader)

if eval_sample_size != -1:
    N = eval_sample_size
else:
    N = total_batches

for i, batch in enumerate(tqdm(eval_dataloader)):
    if i >= N:
        break

    input_ids = batch['input_ids'].to(device)
    attention_masks = batch['attention_mask'].to(device)
    fea0, fea1, fea2 = batch['fea0'].to(device), batch['fea1'].to(device), batch['fea2'].to(device)
    mask0, mask1, mask2 = batch['mask0'].to(device), batch['mask1'].to(device), batch['mask2'].to(device)
    answers = batch['answers']
    
    # Model inference
    res = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        fea0=fea0,
        fea1=fea1,
        fea2=fea2,
        mask0=mask0,
        mask1=mask1,
        mask2=mask2
    )
    
    ans_list.extend(answers)
    res_list.extend(res)

results = {
    "answers": ans_list,
    "results": res_list
}

metrics_open_ended(res_list, ans_list, script_args)

df_results = pd.DataFrame(results)

df_results.to_csv(script_args.results_save_path, index=False)
