from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import ast
import torch
import random
import pandas as pd
from transformers import TrainingArguments, AutoTokenizer, HfArgumentParser
from utils.my_trainer import CustomTrainer
from utils.utils import my_compute_metrics,seed_everything
from typing import Optional
from dataclasses import dataclass, field
from model.qformer import Blip2QformerPathInstruct
from peft import LoraConfig
from datasets import load_dataset, load_from_disk, concatenate_datasets
from utils.data_collator import MyDataCollatorForQFormerPatchInstruct

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
    llm_name: Optional[str] = field(default="mistralai/Mistral-7B-Instruct-v0.2", metadata={"help": "the model name， mistralai/Mistral-7B-Instruct-v0.2, meta-llama/Meta-Llama-3-8B, meta-llama/Llama-2-7b-chat-hf "})
    clip_name: Optional[str] = field(default="conch", metadata={"help": "the model name,  conch / pathclip-base / uni"})
    bert_name: Optional[str] = field(default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", metadata={"help": "the bert name"})
    
    # data
    select_data_num: Optional[int] = field(default=-1, metadata={"help": "the number of training data, -1 mean use all data"})
    dataset_name_list: Optional[str] = field(default="CNX-PathLLM/Pathinstruct,CNX-PathLLM/MultiConversation,CNX-PathLLM/TextbookQAPair", metadata={"help": "CNX-PathLLM/PubMedPath,CNX-PathLLM/CleanedTextData,CNX-PathLLM/TwitterPath,CNX-PathLLM/Pathcap,CNX-PathLLM/TextbookQAPair,CNX-PathLLM/PVQAClean"})
    dataset_local_paths: Optional[str] = field(default="/home/z/zeyugao/dataset/YoutubePathQA/pretrain_data_all", metadata={"help": "the local path for some datasets"})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    data_cache_dir: Optional[str] = field(default="~/.cache", metadata={"help": "the cache dir the dataset and model, /bask/projects/p/phwq4930-gbm/Zeyu/PathVLM/.cache"})
    ckpt_path: Optional[str] = field(default=None, metadata={"help": "ckpt path"})

    # log and save model
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    save_steps: Optional[int] = field(default=500, metadata={"help": "Number of updates steps before two checkpoint saves"})
    save_total_limit: Optional[int] = field(default=5, metadata={"help": "Limits total number of checkpoints."})
    llm_requires_grad: Optional[bool] = field(default=False, metadata={"help": "True or  /output/checkpoint-1400"})
    resume_from_checkpoint: Optional[bool] = field(default=False, metadata={"help": "True or  /output/checkpoint-1400"})
    
    # training hypterparam
    learning_rate: Optional[float] = field(default=2.0e-5, metadata={"help": "the learning rate"})
    train_batch_size: Optional[int] = field(default=40, metadata={"help": "the batch size"})
    eval_batch_size: Optional[int] = field(default=48, metadata={"help": "the batch size"})
    max_seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(default=8, metadata={"help": "the number of gradient accumulation steps"})
    max_steps: Optional[int] = field(default=200_000, metadata={"help": "the number of training steps"})
    num_train_epochs: Optional[int] = field(default=-1, metadata={"help": "the number of training epochs"})
    warmup_ratio: Optional[float] = field(default=0.1, metadata={"help": "the warmup ratio"})
        
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
device = 'cuda'

# set up tokenizer
llm_tokenizer = AutoTokenizer.from_pretrained(script_args.llm_name)
llm_tokenizer.pad_token = llm_tokenizer.eos_token
llm_tokenizer.padding_side = "right"
llm_tokenizer.truncation_side = 'left'

new_tokens = ['<Question>',  '<Answer>', '<Image>']  
num_added_toks = llm_tokenizer.add_tokens(new_tokens)
new_tokens_ids = llm_tokenizer.convert_tokens_to_ids(new_tokens)
print("new_tokens_ids: ", new_tokens_ids)

questions = pd.read_csv('./utils/question_list.csv', header=None)  
questions = questions[0].tolist()

def formatting_func_vqap(examples):
    question = examples["question"]
    answer = examples["answer"]
    question = question.replace("<image>\n", "")
    question = question.replace("<image>", "")
    text = f"<Question> {question}{llm_tokenizer.eos_token}" + f"<Answer> {answer}{llm_tokenizer.eos_token}\n"
    examples["text_input"] = question
    examples["text"] = text
    return examples

def formatting_func_ytb(examples):
    text = examples['conversations'].replace("<image>\n", "").replace("<image>", "")
    question = ast.literal_eval(text[1:-1].split('\n')[0])['value'].replace("\n", "")
    answer = ast.literal_eval(text[1:-1].split('\n')[1])['value'].replace("\n", "")
    text = f"<Question> {question}{llm_tokenizer.eos_token}" + f"<Answer> {answer}{llm_tokenizer.eos_token}\n"
    # text = f"<DES> {answer}{tokenizer.eos_token}\n"
    examples["text_input"] = question
    examples["text"] = text
    return examples

# CNX-PathLLM/MultiConversation
def formatting_func_vmc(examples): # image conversations
    conversation = examples["conversations"]
    conversation = ast.literal_eval(conversation)
    question = ""
    text = ""
    for sentence in conversation:
        sentence['value'] = sentence['value'].replace("<image>\n", "")
        sentence['value'] = sentence['value'].replace("<image>", "")
        if sentence['from'] == 'human':
            text += f"<Question> {sentence['value']}{llm_tokenizer.eos_token}"
            question += sentence['value']
        elif sentence['from'] == 'gpt':
            text += f"<Answer> {sentence['value']}{llm_tokenizer.eos_token}\n"
    examples["text_input"] = question
    examples["text"] = text
    return examples

def formatting_func_ytb(examples):
    text = examples['conversations'].replace("<image>\n", "").replace("<image>", "")
    question = ast.literal_eval(text[1:-1].split('\n')[0])['value'].replace("\n", "")
    answer = ast.literal_eval(text[1:-1].split('\n')[1])['value'].replace("\n", "")
    text = f"<Question> {question}{llm_tokenizer.eos_token}" + f"<Answer> {answer}{llm_tokenizer.eos_token}\n"
    examples["text_input"] = question
    examples["text"] = text
    return examples


if script_args.select_data_num>0:
    split_text = "train[:{}]".format(script_args.select_data_num)
else:
    split_text = "train"
    

train_dataset = []
eval_dataset = []

for dataset_name in script_args.dataset_name_list.split(","):
    one_dataset = load_dataset(dataset_name, split=split_text, cache_dir=script_args.data_cache_dir)
    if dataset_name in ["CNX-PathLLM/Pathinstruct", "CNX-PathLLM/TextbookQAPair"]:
        one_dataset = one_dataset.map(formatting_func_vqap, num_proc=4, remove_columns=["question", "answer"])
    elif dataset_name in ["CNX-PathLLM/MultiConversation"]:
        one_dataset = one_dataset.map(formatting_func_vmc, num_proc=4, remove_columns=["conversations"])
    elif dataset_name in ["CNX-PathLLM/YoutubeInstruct"]:
        one_dataset = one_dataset.map(formatting_func_ytb, num_proc=4, remove_columns=['id','conversations'])
    train_dataset.append(one_dataset)

train_dataset = concatenate_datasets(train_dataset)

if eval_dataset == []:
    eval_dataset = None
else:
    eval_dataset = concatenate_datasets(eval_dataset)

model = Blip2QformerPathInstruct(
                                    clip_name = script_args.clip_name,
                                    num_query_token = 16,
                                    cross_attention_freq = 2,
                                    pretrain_name = script_args.bert_name,
                                    llm_requires_grad = script_args.llm_requires_grad, 
                                    load_in_8bit = script_args.load_in_8bit, 
                                    load_in_4bit = script_args.load_in_4bit, 
                                    llm_name = script_args.llm_name, 
                                    trust_remote_code = script_args.trust_remote_code, 
                                    token = script_args.token, 
                                    llm_tokenizer = llm_tokenizer,
                                    image_token_id = new_tokens_ids[-1],
                                    data_cache_dir = script_args.data_cache_dir
                                )

if script_args.ckpt_path is not None:
    model.load_state_dict(torch.load(script_args.ckpt_path, map_location=device), strict=False)
    model = model.to(torch.bfloat16)
    print("load qformer from: {}".format(script_args.ckpt_path))

print("output dir is set to: {}".format(script_args.output_dir))

training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.train_batch_size,
    per_device_eval_batch_size=script_args.eval_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    # gradient_checkpointing=True,
    learning_rate=script_args.learning_rate,
    # lr_scheduler_type="cosine",
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    bf16=True,
    warmup_ratio=script_args.warmup_ratio,
    evaluation_strategy=script_args.evaluation_strategy,
    eval_steps=script_args.eval_steps,
    logging_first_step=True,
    remove_unused_columns=False,
    label_names=["labels"]
)

if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None

data_collator = MyDataCollatorForQFormerPatchInstruct(image_processor=model.image_processor, tokenizer=llm_tokenizer)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    max_seq_length=script_args.max_seq_length,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field=script_args.dataset_text_field,
    peft_config=peft_config,
    tokenizer=llm_tokenizer,
    data_collator=data_collator,
    compute_metrics=my_compute_metrics,
)


trainer.train(resume_from_checkpoint=script_args.resume_from_checkpoint)