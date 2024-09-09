from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import torch
import ast
import re
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, HfArgumentParser
from torch.utils.data import DataLoader
from utils.utils import seed_everything
from typing import Optional
from dataclasses import dataclass, field
from peft import LoraConfig, get_peft_model
from model.my_model import PPathVLM
from model.qformer import Blip2QformerPathInstruct
from datasets import load_dataset, concatenate_datasets, load_from_disk
from utils.data_collator import MyDataCollatorForPPathVLMTest, MyDataCollatorForQFormerPatchInstruct

from utils.eval_utils import calculate_prf_score, compute_bleu_scores, split_sentence, compute_cider_scores, compute_spice_scores
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
    
    bert_name: Optional[str] = field(default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", metadata={"help": "the bert name"})
    adaptor: Optional[str] = field(default="linear", metadata={"help": "the adaptor name, linear or qformer"})
    
    
    max_seq_length: Optional[int] = field(default=256, metadata={"help": "Input sequence length"})
    llm_requires_grad: Optional[bool] = field(default=False, metadata={"help": "True or  /output/checkpoint-1400"})
    
    # data
    # dataset_name_list: Optional[str] = field(default="CNX-PathLLM/Pathcap,CNX-PathLLM/PubMedPath,CNX-PathLLM/TwitterPath,CNX-PathLLM/CleanedTextData", metadata={"help": ""})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    dataset_local_paths: Optional[str] = field(default=None, metadata={"help": "the local path for some datasets"})
    data_cache_dir: Optional[str] = field(default="/bask/homes/a/asiw9691/PathVLM/.cache", metadata={"help": "the cache dir the dataset and model, /bask/projects/p/phwq4930-gbm/Zeyu/PathVLM/.cache"})
    
    # eval
    batch_size: Optional[int] = field(default=16, metadata={"help": "batch_size"})
    test_img_dir: Optional[str] = field(default='/bask/homes/a/asiw9691/PathVLM/source/PathLLM/test_images/*.jpeg', metadata={"help": "test sample images dic"})
    ckpt_path: Optional[str] = field(default="/bask/projects/p/phwq4930-renal-canc/Zeyu/PathVLM/source/PathLLM/output/Conch_Llama3_Patch_VQA/ckpt10500.bin", metadata={"help": "ckpt path"})
    
    #lora
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters, 4 to 64"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters, 16 to 128"})


def setup_tokenizer(script_args):
    # set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.llm_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = 'left'

    new_tokens = ['<Question>',  '<Answer>', '<Image>']
    num_added_toks = tokenizer.add_tokens(new_tokens)
    new_tokens_ids = tokenizer.convert_tokens_to_ids(new_tokens)
    print("new_tokens_ids: ", new_tokens_ids)
    return tokenizer, new_tokens_ids

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
seed_everything(script_args.seed)
tokenizer, new_tokens_ids = setup_tokenizer(script_args)


def formatting_func_pathvqa(examples):
    question = examples["question"]
    answer = examples["answer"]
    question = question.replace("<image>\n", "")
    question = question.replace("<image>", "")
    if answer in ["yes","no"]:
        examples["answer_type"] = "CLOSED"
        # question += " answer yes or no directly!"
    else:
        examples["answer_type"] = "OPEN"
    text = f"<Question> {question}{tokenizer.eos_token}" # + f"<Answer>"
    examples["text_input"] = question
    examples["answer"] = answer
    examples["text"] = text
    return examples

def formatting_func_quiltvqa(examples):
    question = examples["question"]
    answer = examples["answer"]
    # if examples["answer_type"] == "CLOSED":
        # question += " answer yes or no directly!"
    text = f"<Question> {question}{tokenizer.eos_token}" # + f"<Answer>"
    examples["text_input"] = question
    examples["answer"] = answer
    examples["text"] = text
    return examples

def setup_datasets(script_args):
# set up datasets
    dataset_quilt = load_dataset("wisdomik/Quilt_VQA", split="train", cache_dir=script_args.data_cache_dir)
    dataset_quilt = dataset_quilt.map(formatting_func_quiltvqa, num_proc=4, remove_columns=["question", "context"])
    closed_dataset_quilt = dataset_quilt.filter(lambda example: example['answer_type'] == 'CLOSED').remove_columns('answer_type')
    open_dataset_quilt = dataset_quilt.filter(lambda example: example['answer_type'] == 'OPEN').remove_columns('answer_type')

    dataset_pvqa = load_dataset("CNX-PathLLM/PVQAClean", split="train", cache_dir=script_args.data_cache_dir)
    dataset_pvqa = dataset_pvqa.map(formatting_func_pathvqa, num_proc=4, remove_columns=["question"])
    closed_dataset_pvqa = dataset_pvqa.filter(lambda example: example['answer_type'] == 'CLOSED').remove_columns('answer_type')
    open_dataset_pvqa = dataset_pvqa.filter(lambda example: example['answer_type'] == 'OPEN').remove_columns('answer_type')

    dataset = [closed_dataset_quilt, open_dataset_quilt, closed_dataset_pvqa, open_dataset_pvqa]

    dataset_name = ['closed_dataset_quilt','open_dataset_quilt','closed_dataset_pvqa','open_dataset_pvqa']

    return dataset, dataset_name

def setup_model(script_args):
    
    if script_args.adaptor == 'qformer':
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
                                            llm_tokenizer = tokenizer,
                                            image_token_id = new_tokens_ids[-1],
                                            data_cache_dir = script_args.data_cache_dir
                                        )
    else:
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

    model.load_state_dict(torch.load(script_args.ckpt_path, map_location=device))
    model.to(device)
    
    if script_args.adaptor == 'qformer':
        model.to(torch.bfloat16)

    # model.print_llm_parameters()
    # model.print_vision_parameters()
    return model

def setup_dataloader(dataset, tokenizer, data_collator, script_args):
    if script_args.adaptor == 'qformer':
        remove_columns = []
    else:
        remove_columns=['text']
    tokenized_dataset = dataset.map(
                    tokenizer,
                    batched=False,
                    remove_columns=remove_columns,
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

def predict_and_save_open_ended(eval_dataloader, model, script_args):
    # todo CIDEr and SPICE
    open_ques_pre = []
    open_ques_rec = []
    open_ques_f1 = []
    open_bleu_score = []
    open_candidate = []
    open_reference = []

    total_batches = len(eval_dataloader)

    N = total_batches

    for i, batch in enumerate(tqdm(eval_dataloader)):
        if i >= N:
            break

    # for batch in tqdm(eval_dataloader):

        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        answers = batch['answers']
        p_num = batch["patch_num"]
        text_input = batch["text_input"]

        # 执行模型推断
        res = model.generate(input_ids=input_ids,
                            attention_mask=attention_masks,
                            patch_num=p_num,
                            text_input=text_input,
                            image=images)
        
        for i in range(len(answers)):
                if  len(split_sentence(answers[i], 1)) == 0:
                    continue
                open_candidate.append(res[i])
                open_reference.append(answers[i])


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

def predict_and_save_close_ended(eval_dataloader, model, script_args):
    total_num = 0
    correct_num = 0
    close_candidate = []
    close_reference = []

    total_batches = len(eval_dataloader)

    N = total_batches

    for i, batch in enumerate(tqdm(eval_dataloader)):
        if i >= N:
            break

    # for batch in tqdm(eval_dataloader):

        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        answers = batch['answers']
        p_num = batch["patch_num"]
        text_input = batch["text_input"]

        # 执行模型推断
        res = model.generate(input_ids=input_ids,
                            attention_mask=attention_masks,
                            patch_num=p_num,
                            text_input=text_input,
                            image=images)
        
        for i in range(len(answers)):
                if  len(split_sentence(answers[i], 1)) == 0:
                    continue
                close_candidate.append(res[i])
                close_reference.append(answers[i])


    # calculate accuracy for close ended problem
    for i in range(len(close_reference)):
        first_word_ref = re.findall(r'\b\w+\b', close_reference[i])[0].lower()
        first_word_cand = re.findall(r'\b\w+\b', close_candidate[i])[1].lower()
        if first_word_cand in ['yes','no'] and first_word_ref in ['yes','no']:
            total_num += 1
            if first_word_cand == first_word_ref:
                correct_num += 1

    acc = correct_num/total_num

    print("closed question accuracy: {}\n".format(acc))

    output_path = script_args.ckpt_path.replace('.bin', '.txt')

    with open(output_path, 'a') as file:
        file.write("closed question accuracy: {}\n".format(acc))

    print("Results have been written to {}".format(output_path))

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
        text_list.append(f"<Question> What is the final pathological diagnosis for this image?{tokenizer.eos_token}")

    patch_list = torch.stack(patch_list) # [448x448]
        
    input_dic = tokenizer(text_list, return_tensors="pt")
    input_dic["image"] = patch_list
    input_dic["patch_num"] = num_list
    input_dic["text_input"] = text_list

    res = model.generate(input_ids=input_dic["input_ids"].to(device),
                     attention_mask=input_dic["attention_mask"].to(device),
                     patch_num=input_dic["patch_num"],
                     text_input=input_dic["text_input"],
                     image=input_dic["image"].to(device))
    
    for i in range(len(res)):
        print("{}: {} \n".format(image_paths[i], res[i]))

        
dataset, dataset_name = setup_datasets(script_args)
model = setup_model(script_args)

if script_args.adaptor == 'qformer':
    data_collator = MyDataCollatorForQFormerPatchInstruct(image_processor=model.image_processor, tokenizer=tokenizer, test=True)
else:
    data_collator = MyDataCollatorForPPathVLMTest(tokenizer=tokenizer, image_processor=model.image_processor)

for i in range(len(dataset)):
    eval_dataloader = setup_dataloader(dataset[i], tokenizer, data_collator, script_args)
    if 'open' in dataset_name[i]:
        predict_and_save_open_ended(eval_dataloader, model, script_args)
    else:
        predict_and_save_close_ended(eval_dataloader, model, script_args)

# visualize some answers for samples
test_some_samples(script_args.test_img_dir, model, data_collator, tokenizer)