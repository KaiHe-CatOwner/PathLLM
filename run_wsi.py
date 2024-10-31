from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import ast
import random
import pandas as pd
from transformers import TrainingArguments, AutoTokenizer, HfArgumentParser
from utils.my_trainer import CustomTrainer
from utils.utils import my_compute_metrics,seed_everything
from typing import Optional
from dataclasses import dataclass, field
from model.my_model import PPathVLM, WPathVLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, concatenate_datasets, load_from_disk
from utils.data_collator import MyDataCollatorForWPathVLM

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    # system config
    gpu: Optional[str] = field(default="2", metadata={"help": "gpu"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    seed: Optional[int] = field(default=42, metadata={"help": "seed"})

    # model
    llm_name: Optional[str] = field(default="meta-llama/Meta-Llama-3.1-8B", metadata={"help": "the model name， mistralai/Mistral-7B-Instruct-v0.2, meta-llama/Meta-Llama-3-8B"})
    
    # data
    select_data_num: Optional[int] = field(default=-1, metadata={"help": "the number of training data， -1 mean use all data"})
    dataset_name_list: Optional[str] = field(default="CNX-PathLLM/TCGA-WSI-Description, CNX-PathLLM/GTEx-WSI-Description")
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    data_cache_dir: Optional[str] = field(default="~/.cache", metadata={"help": "the cache dir the dataset and model, /bask/projects/p/phwq4930-gbm/Zeyu/PathVLM/.cache"})
    data_local_dir: Optional[str] = field(default=None, metadata={"help": "if not None, load from local"})
    fea_root: Optional[str] = field(default="/bask/homes/a/asiw9691/PathVLM/WSI_Dataset/Conch/", metadata={"help": "the root path for WSI feature"})
    ckpt_path: Optional[str] = field(default=None, metadata={"help": "ckpt path"})

    # log and save model
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    logging_steps: Optional[int] = field(default=5, metadata={"help": "the number of logging steps"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    warmup_steps: Optional[int] = field(default=20, metadata={"help": "the number of warmup steps"})
    save_steps: Optional[int] = field(default=500, metadata={"help": "Number of updates steps before two checkpoint saves"})
    save_total_limit: Optional[int] = field(default=1, metadata={"help": "Limits total number of checkpoints."})
    
    llm_requires_grad: Optional[bool] = field(default=False, metadata={"help": "True or  /output/checkpoint-1400"})
    resume_from_checkpoint: Optional[bool] = field(default=False, metadata={"help": "True or  /output/checkpoint-1400"})
    
    # training hypterparam
    learning_rate: Optional[float] = field(default=2.0e-5, metadata={"help": "the learning rate"})
    train_batch_size: Optional[int] = field(default=40, metadata={"help": "the batch size"})
    eval_batch_size: Optional[int] = field(default=48, metadata={"help": "the batch size"})
    max_seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(default=8, metadata={"help": "the number of gradient accumulation steps"})
    num_train_epochs: Optional[int] = field(default=5, metadata={"help": "the number of training epochs"})

    # WSI hyperparam
    n_level: Optional[int] = field(default=3, metadata={"help": "the number of herachical levels for WSI embedding"})
    embed_dim: Optional[int] = field(default=512, metadata={"help": "embedding dimension of each patch, conch: 512, gmm: 2*d+1"})
    agg_strategy: Optional[str] = field(default='abmil', metadata={"help": "the strategy for WSI aggregation, sample, kmeans, gmm, abmil, longnet"})
    n_heads: Optional[str] = field(default='32,16,8', metadata={"help": "the number of attention heads for WSI aggregation, for sample and abmil"})
    
    # eval
    evaluation_strategy: Optional[str] = field(default="steps", metadata={"help": "epoch, step"})
    eval_steps: Optional[int] = field(default=100000, metadata={"help": "eval_steps"})

    # unused
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default="mistral-7b-finetuned-ultrachat", metadata={"help": "The name of the model on HF Hub"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
seed_everything(script_args.seed)

# os.environ["WANDB_MODE"] = "offline"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = script_args.gpu

# set up tokenizer
tokenizer = AutoTokenizer.from_pretrained(script_args.llm_name)
#tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token = "<|finetune_right_pad_id|>"
tokenizer.padding_side = 'right'
tokenizer.truncation_side = 'right'

new_tokens = ['<|Question|>', '<|Prompt|>', '<|Answer|>', '<|Image|>']
num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
# num_added_toks = tokenizer.add_tokens(new_tokens)
new_tokens_ids = tokenizer.convert_tokens_to_ids(new_tokens)
print("new_tokens_ids: ", new_tokens_ids)

questions = pd.read_csv('./utils/question_wsi_list.csv', header=None)  
questions = questions[0].tolist()

def formatting_func_des(examples):
    question = random.choice(questions)
    answer = examples["description"]
    text = f"<|Question|>{question}" + f"<|Answer|>{answer}{tokenizer.eos_token}"
    examples["text"] = text
    return examples

def formatting_func_qa_open(examples):
    question = examples["question"]
    answer = examples["answer"]
    text = f"<|Question|>{question}" + f"<|Answer|>{answer}{tokenizer.eos_token}"
    examples["text"] = text
    return examples

def formatting_func_qa_close(examples):
    question = examples["question"]
    answer = examples["answer"]
    if answer.lower() in ['yes', 'no']:
        prompt = f" Please provide only the answer (either Yes or No) for the following statement. Do not include any explanations or additional text. Just give Yes or No."
    else:
        prompt = f" Please provide only the answer (for example, A. [Answer Text], B. [Answer Text], etc.) for the following question. Do not include any explanations or additional text. Just give the letter followed by the corresponding answer."
    
    text = f"<|Question|>{question}<|Prompt|>{prompt}" + f"<|Answer|>{answer}{tokenizer.eos_token}"
    examples["text"] = text
    return examples

if script_args.select_data_num>0:
    split_text = "train[:{}]".format(script_args.select_data_num)
else:
    split_text = "train"

# if script_args.data_local_dir is None:
dataset = []

for dataset_name in script_args.dataset_name_list.split(","):
    columns_to_remove = ['slide_id']
    one_dataset = load_dataset(dataset_name, split=split_text, cache_dir=script_args.data_cache_dir)
    if 'project' in one_dataset.column_names:
        columns_to_remove.append('project')
    elif 'site' in one_dataset.column_names:
        columns_to_remove.append('site')

    if 'QA' in dataset_name:  # for QA instruction dataset
        columns_to_remove += ['question', 'answer']
        if 'Open' in dataset_name: # for OpenQA instruction dataset
            one_dataset = one_dataset.map(formatting_func_qa_open, num_proc=20, remove_columns=columns_to_remove)
        else: # for CloseQA instruction dataset
            one_dataset = one_dataset.map(formatting_func_qa_close, num_proc=20, remove_columns=columns_to_remove)
    else:
        columns_to_remove += ['description']
        one_dataset = one_dataset.map(formatting_func_des, num_proc=20, remove_columns=columns_to_remove)
    dataset.append(one_dataset)

dataset = concatenate_datasets(dataset)

# dataset = dataset.train_test_split(test_size=0.05)
# eval_dataset = dataset['test']
# train_dataset = dataset['train']
train_dataset = dataset
eval_dataset = None

# else:
#     dataset = load_from_disk(script_args.data_local_dir)
#     dataset = dataset.map(formatting_func, num_proc=20, remove_columns=['label', 'slide_id', 'project'])
#     train_folds = [dataset[f'fold_{i}'] for i in range(10) if i != script_args.eval_fold_index]
#     train_dataset = concatenate_datasets(train_folds)
#     eval_dataset = dataset['fold_{}'.format(script_args.eval_fold_index)]

# df_indices = pd.read_csv(script_args.dataset_split)

# train_indices = df_indices[df_indices['fold'] != (script_args.test_split_fold-1)]['index'].tolist()
# test_indices = df_indices[df_indices['fold'] == (script_args.test_split_fold-1)]['index'].tolist()

# train_dataset = dataset.select(train_indices)
# eval_dataset = dataset.select(test_indices)
print(train_dataset)
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

model.print_parameter_counts()
model.print_llm_parameters()

print("output dir is set to: {}".format(script_args.output_dir))

training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.train_batch_size,
    per_device_eval_batch_size=script_args.eval_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    # gradient_checkpointing=True,
    learning_rate=script_args.learning_rate,
    lr_scheduler_type="constant_with_warmup",
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    bf16=True,
    warmup_steps=script_args.warmup_steps,
    evaluation_strategy=script_args.evaluation_strategy,
    eval_steps=script_args.eval_steps,
    logging_first_step=True,
    remove_unused_columns=False,
    label_names=["labels"]
)

if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,  # Use a moderate rank
        lora_alpha=script_args.peft_lora_alpha,  # Scaling factor
        bias="none",  # No bias adaptation
        task_type="CAUSAL_LM",  # For causal language modeling tasks
        # lora_dropout=0.1,  # Use dropout for regularization
        # target_modules=["q_proj", "v_proj"],  # Focus on key attention components
        # init_lora_weights="pissa"  # Use random initialization
    )
    model.llm = get_peft_model(model.llm, peft_config)
    model.llm.print_trainable_parameters()
else:
    peft_config = None

if script_args.ckpt_path is not None:
    model.load_state_dict(torch.load(script_args.ckpt_path, map_location=device), strict=False)
    # model = model.to(torch.bfloat16)
    print("load pre-trained model from: {}".format(script_args.ckpt_path))
    model.print_llm_parameters()

data_collator = MyDataCollatorForWPathVLM(tokenizer=tokenizer, 
                                        fea_root=script_args.fea_root, 
                                        fea_dim=script_args.embed_dim, 
                                        n_level=script_args.n_level,
                                        n_heads=list(map(int, script_args.n_heads.split(','))),
                                        agg_strategy=script_args.agg_strategy)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    max_seq_length=script_args.max_seq_length,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field=script_args.dataset_text_field,
    peft_config=None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=my_compute_metrics,
)

trainer.train(resume_from_checkpoint=script_args.resume_from_checkpoint)