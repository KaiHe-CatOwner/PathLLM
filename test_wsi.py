from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import random
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, HfArgumentParser
from utils.utils import seed_everything
from typing import Optional
from dataclasses import dataclass, field
from utils.data_collator import MyDataCollatorForWPathVLMTest
from datasets import load_dataset, concatenate_datasets, load_from_disk
from model.my_model import WPathVLM

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
    
    # WSI hyperparameters
    n_heads: Optional[int] = field(default=2, metadata={"help": "Number of attention heads for WSI aggregation"})
    n_level: Optional[int] = field(default=3, metadata={"help": "Number of hierarchical levels for WSI embedding"})
    embed_dim: Optional[int] = field(default=512, metadata={"help": "Embedding dimension of each patch"})
    
    # Evaluation
    batch_size: Optional[int] = field(default=4, metadata={"help": "Batch size"})
    ckpt_path: Optional[str] = field(default=None, metadata={"help": "Checkpoint path"})
    shuffle: Optional[bool] = field(default=False, metadata={"help": "shuffle eval_dataloader or not"})
    eval_sample_size: Optional[int] = field(default=-1, metadata={"help": "-1 indicate evaluating on all"})

device = 'cuda'

def formatting_func(examples):
    # question = random.choice(questions)
    # answer = examples["label"]
    text = f"<|Describe|>"
    examples["text"] = text
    examples["answer"] = examples["label"]
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
new_tokens = ['<|Question|>',  '<|Answer|>', '<|Describe|>', '<|Image|>']
num_added_toks = tokenizer.add_tokens(new_tokens)
new_tokens_ids = tokenizer.convert_tokens_to_ids(new_tokens)
print("New tokens IDs:", new_tokens_ids)

# Determine data split
split_text = f"train[:{script_args.select_data_num}]" if script_args.select_data_num > 0 else "train"

# Load dataset
if script_args.data_local_dir is None:
    datasets = [load_dataset(name, split=split_text, cache_dir=script_args.data_cache_dir) for name in script_args.dataset_name_list.split(",")]
    dataset = concatenate_datasets(datasets)
    dataset = dataset.train_test_split(test_size=0.05)
    eval_dataset = dataset['test']
else:
    dataset = load_from_disk(script_args.data_local_dir)
    eval_dataset = dataset[f'fold_{script_args.eval_fold_index}']

# Format and tokenize dataset
if script_args.eval_sample_size > 0:
    random_indices = random.sample(range(len(eval_dataset)), script_args.eval_sample_size)
    eval_dataset = eval_dataset.select(random_indices)

eval_dataset = eval_dataset.map(formatting_func, num_proc=20, remove_columns=['label', 'slide_id', 'project'])
tokenized_dataset = eval_dataset.map(tokenize, batched=False, num_proc=4, remove_columns=['text'], 
                                     batch_size=script_args.batch_size, input_columns=['text'])

# Load model
model = WPathVLM(
    script_args.llm_requires_grad, 
    script_args.load_in_8bit, 
    script_args.load_in_4bit, 
    script_args.llm_name, 
    script_args.trust_remote_code,  # False
    script_args.token,  # True
    tokenizer,
    new_tokens_ids[-1],
    n_heads=script_args.n_heads, 
    n_level=script_args.n_level, 
    embed_dim=script_args.embed_dim,
    data_cache_dir=script_args.data_cache_dir
)
model.load_state_dict(torch.load(script_args.ckpt_path, map_location=device))
model.to(device)

# Prepare data loader
data_collator = MyDataCollatorForWPathVLMTest(tokenizer=tokenizer, fea_dim=512, n_level=3)
dataloader_params = {"batch_size": script_args.batch_size, "collate_fn": data_collator, "shuffle": script_args.shuffle}
eval_dataloader = DataLoader(tokenized_dataset, **dataloader_params)

# Evaluate the model
ans_list, res_list = [], []

for batch in tqdm(eval_dataloader):
    input_ids = batch['input_ids'].to(device)
    attention_masks = batch['attention_mask'].to(device)
    fea1, fea2, fea3 = batch['fea1'].to(device), batch['fea2'].to(device), batch['fea3'].to(device)
    mask1, mask2, mask3 = batch['mask1'].to(device), batch['mask2'].to(device), batch['mask3'].to(device)
    answers = batch['answers']
    
    # Model inference
    res = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        fea1=fea1,
        fea2=fea2,
        fea3=fea3,
        mask1=mask1,
        mask2=mask2,
        mask3=mask3
    )
    
    ans_list.extend(answers)
    res_list.extend(res)

results = {
    "answers": ans_list,
    "results": res_list
}

df_results = pd.DataFrame(results)

df_results.to_csv(script_args.results_save_path, index=False)
