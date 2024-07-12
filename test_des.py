from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import torch
import ast
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, HfArgumentParser
from torch.utils.data import DataLoader
from utils.utils import seed_everything
from typing import Optional
from dataclasses import dataclass, field
from model.my_model import PPathVLM
from datasets import load_dataset, concatenate_datasets, load_from_disk
from utils.data_collator import MyDataCollatorForPPathVLMTest

from utils.eval_utils import calculate_prf_score, compute_bleu_scores
device = 'cuda'

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    # system config
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    seed: Optional[int] = field(default=2024, metadata={"help": "seed"})

    # model
    llm_name: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B-Instruct", metadata={"help": "the model name，  meta-llama/Llama-2-7b-chat-hf "})
    clip_name: Optional[str] = field(default="conch", metadata={"help": "the model name，  conch / pathclip-base / uni"})
    max_seq_length: Optional[int] = field(default=256, metadata={"help": "Input sequence length"})
    llm_requires_grad: Optional[bool] = field(default=False, metadata={"help": "True or  /output/checkpoint-1400"})
    
    # data
    dataset_name_list: Optional[str] = field(default="CNX-PathLLM/Pathcap,CNX-PathLLM/PubMedPath,CNX-PathLLM/TwitterPath,CNX-PathLLM/CleanedTextData", metadata={"help": ""})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    dataset_local_paths: Optional[str] = field(default=None, metadata={"help": "the local path for some datasets"})
    data_cache_dir: Optional[str] = field(default="/home/z/zeyugao/.cache", metadata={"help": "the cache dir the dataset and model, /bask/projects/p/phwq4930-gbm/Zeyu/PathVLM/.cache"})
    
    # eval
    batch_size: Optional[int] = field(default=16, metadata={"help": "batch_size"})
    test_img_dir: Optional[str] = field(default='/home/z/zeyugao/PathLLM/test_images/*.jpeg', metadata={"help": "test sample images dic"})
    ckpt_path: Optional[str] = field(default="/home/z/zeyugao/PathLLM/output/Conch_Llama3_PatchInstruct_IT_nytb_des/ckpt_1000.bin", metadata={"help": "ckpt path"})

def test_some_samples(test_image_dir, model, data_collator, tokenizer):
    # test several sample images

    image_paths = glob(test_image_dir)

    patch_list = []
    num_list = []
    text_list = []

    for image_path in image_paths:
        image = Image.open(image_path)
        image = data_collator._resize_image(image)
        patches = data_collator._crop_image(image) # [448x448]
        patches = [model.image_processor(patch) for patch in patches]
        num_list.append(len(patches))
        patch_list += patches
        text_list.append(f"<DES>")

    patch_list = torch.stack(patch_list) # [448x448]
        
    input_dic = tokenizer(text_list, return_tensors="pt")
    input_dic["image"] = patch_list
    input_dic["patch_num"] = num_list

    res = model.generate(input_ids=input_dic["input_ids"].to(device),
                     attention_mask=input_dic["attention_mask"].to(device),
                     patch_num=input_dic["patch_num"],
                     image=input_dic["image"].to(device))
    
    for i in range(len(res)):
        print("Description for {}: {} \n".format(image_paths[i], res[i]))

def setup_tokenizer(script_args):
    # set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.llm_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = 'left'

    new_tokens = ['<Question>',  '<Answer>', '<DES>', '<Image>']
    num_added_toks = tokenizer.add_tokens(new_tokens)
    new_tokens_ids = tokenizer.convert_tokens_to_ids(new_tokens)
    print("new_tokens_ids: ", new_tokens_ids)
    return tokenizer, new_tokens_ids

def formatting_func_itp(examples):
    answer = examples["txt"]
    text = f"<DES>"
    examples["text"] = text
    examples["answer"] = answer
    return examples

def formatting_func_ytb(examples):
    text = examples['conversations'].replace("<image>\n", "").replace("<image>", "")
    question = ast.literal_eval(text[1:-1].split('\n')[0])['value'].replace("\n", "")
    answer = ast.literal_eval(text[1:-1].split('\n')[1])['value'].replace("\n", "")
    text = f"<DES>"
    examples["text"] = text
    examples["answer"] = answer
    return examples

def setup_datasets(script_args):
# set up datasets
    dataset = []
    eval_dataset = None

    for dataset_name in script_args.dataset_name_list.split(","):
        one_dataset = load_dataset(dataset_name, split="train", cache_dir=script_args.data_cache_dir)
        one_dataset = one_dataset.rename_column('jpg', 'image')
        one_dataset = one_dataset.map(formatting_func_itp, num_proc=4, remove_columns=['txt','__key__', '__url__'])
        dataset.append(one_dataset)

    if script_args.dataset_local_paths != None:
        for dataset_name in script_args.dataset_local_paths.split(","):
            one_dataset = load_from_disk(dataset_name)
            one_dataset = one_dataset.map(formatting_func_ytb, num_proc=4, remove_columns=['id','conversations'])
            dataset.append(one_dataset)

    dataset = concatenate_datasets(dataset)
    dataset = dataset.train_test_split(test_size=0.001)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    return train_dataset, eval_dataset

def setup_model(script_args):
    model = PPathVLM(llm_requires_grad = script_args.llm_requires_grad, 
                        clip_name = script_args.clip_name, 
                        load_in_8bit = script_args.load_in_8bit, 
                        load_in_4bit = script_args.load_in_4bit, 
                        llm_name = script_args.llm_name, 
                        trust_remote_code = script_args.trust_remote_code, 
                        token = script_args.token, 
                        tokenizer = tokenizer,
                        image_token_id = new_tokens_ids[-1],
                        data_cache_dir = script_args.data_cache_dir)

    model.load_state_dict(torch.load(script_args.ckpt_path, map_location=device))
    model.to(device)
    return model

def setup_dataloader(dataset, tokenizer, data_collator, script_args):
    tokenized_dataset = dataset.map(
                    tokenizer,
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

    dataloader = DataLoader(tokenized_dataset, **dataloader_params)

    return dataloader

def predict_and_save_openEnded(eval_dataloader, model, script_args):
    # todo CIDEr and SPICE
    open_ques_pre = []
    open_ques_rec = []
    open_ques_f1 = []
    open_candidate = []
    open_reference = []

    for batch in tqdm(eval_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        answers = batch['answers']
        p_num = batch["patch_num"]

        # 执行模型推断
        res = model.generate(input_ids=input_ids,
                            attention_mask=attention_masks,
                            patch_num=p_num,
                            image=images)
        for i in range(len(answers)):
                open_candidate.append(res[i])
                open_reference.append(answers[i])
            
    # calculate f1 score for open ended problem
    for i in range(len(open_reference)):
        precision, recall, f1_score = calculate_prf_score(open_candidate[i], open_reference[i])
        open_ques_pre.append(precision)
        open_ques_rec.append(recall)
        open_ques_f1.append(f1_score)
        
    open_bleu_score = compute_bleu_scores(open_candidate, open_reference, avg=True)

    open_ques_pre = np.mean(open_ques_pre)
    open_ques_rec = np.mean(open_ques_rec)
    open_ques_f1 = np.mean(open_ques_f1)

    output_path = script_args.ckpt_path.replace('.bin', '.txt')

    with open(output_path, 'w') as file:
        file.write("open question macro f1_score: {}\n".format(open_ques_f1))
        file.write("open question macro precision: {}\n".format(open_ques_pre))
        file.write("open question macro recall: {}\n".format(open_ques_rec))
        file.write("open question bleu score: {}\n".format(open_bleu_score))

    print("Results have been written to {}".format(output_path))

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    seed_everything(script_args.seed)
    _, eval_dataset = setup_datasets(script_args)

    tokenizer, new_tokens_ids = setup_tokenizer(script_args)
    model = setup_model(script_args)
    data_collator = MyDataCollatorForPPathVLMTest(tokenizer=tokenizer, image_processor=model.image_processor)

    eval_dataloader = setup_dataloader(eval_dataset, tokenizer, data_collator, script_args)
    # visualize some answers for samples
    test_some_samples(script_args.test_img_dir, model, data_collator, tokenizer)

    predict_and_save_openEnded(eval_dataloader, model, script_args)