import random
import pandas as pd

ques_patch = pd.read_csv('./utils/question_list.csv', header=None)  
ques_patch = ques_patch[0].tolist()

ques_wsi = pd.read_csv('./utils/question_wsi_list.csv', header=None)  
ques_wsi = ques_wsi[0].tolist()

def patch_formatting_itp(examples, tokenizer):
    question = random.choice(ques_patch)
    answer = examples["txt"]
    text = f"<|Question|>{question}" + f"<|Answer|>{answer}{tokenizer.eos_token}"
    # text = f"<DES> {answer}{tokenizer.eos_token}"
    examples["text"] = text
    return examples

def patch_formatting_vqap(examples, tokenizer):
    question = examples["question"]
    answer = examples["answer"]
    question = question.replace("<image>\n", "")
    question = question.replace("<image>", "")
    text = f"<|Question|>{question}" + f"<|Answer|>{answer}{tokenizer.eos_token}"
    examples["text"] = text
    return examples

# CNX-PathLLM/MultiConversation
# [{'from': 'human', 'value': 'What are the key features of this image that suggest chronic pancreatitis?'},
# {'from': 'gpt', 'value': 'The presence of duct dilatation, fibrosis, and pancreatic tissue necrosis are indicative of chronic pancreatitis.}]
def patch_formatting_vmc(examples, tokenizer): # image conversations
    conversation = examples["conversations"]
    conversation = ast.literal_eval(conversation)
    text = ""
    for sentence in conversation:
        sentence['value'] = sentence['value'].replace("<image>\n", "")
        sentence['value'] = sentence['value'].replace("<image>", "")
        if sentence['from'] == 'human':
            text += f"<|Question|>{sentence['value']}"
        elif sentence['from'] == 'gpt':
            text += f"<|Answer|>{sentence['value']}{tokenizer.eos_token}"
    examples["text"] = text
    return examples

def patch_formatting_ytb(examples, tokenizer):
    text = examples['conversations'].replace("<image>\n", "").replace("<image>", "")
    question = ast.literal_eval(text[1:-1].split('\n')[0])['value'].replace("\n", "")
    answer = ast.literal_eval(text[1:-1].split('\n')[1])['value'].replace("\n", "")
    text = f"<|Question|>{question}" + f"<|Answer|>{answer}{tokenizer.eos_token}"
    # text = f"<DES>{answer}{tokenizer.eos_token}"
    examples["text"] = text
    return examples

def wsi_formatting_des(examples, tokenizer):
    question = random.choice(ques_wsi)
    answer = examples["description"]
    text = f"<|Question|>{question}" + f"<|Answer|>{answer}{tokenizer.eos_token}"
    examples["text"] = text
    examples["text_input"] = question
    return examples

def wsi_formatting_qa_open(examples, tokenizer):
    question = examples["question"]
    answer = examples["answer"]
    text = f"<|Question|>{question}" + f"<|Answer|>{answer}{tokenizer.eos_token}"
    examples["text"] = text
    examples["text_input"] = question
    return examples

def wsi_formatting_qa_close(examples, tokenizer, prompt_tag=False):
    question = examples["question"]
    answer = examples["answer"]
    if answer.lower() in ['yes', 'no']:
        prompt = f" Please provide only the answer (either Yes or No) for the following statement. Do not include any explanations or additional text. Just give Yes or No."
    else:
        prompt = f" Please provide only the answer (for example, A. [Answer Text], B. [Answer Text], etc.) for the following question. Do not include any explanations or additional text. Just give the letter followed by the corresponding answer."
    
    if prompt_tag:
        text = f"<|Question|>{question}<|Prompt|>{prompt}" + f"<|Answer|>{answer}{tokenizer.eos_token}"
    else:
        text = f"<|Question|>{question}" + f"<|Answer|>{answer}{tokenizer.eos_token}"

    examples["text"] = text
    examples["text_input"] = question
    return examples

def wsi_formatting_des_test(examples, tokenizer):
    question = random.choice(ques_wsi)
    answer = examples["description"]
    text = f"<|Question|>{question}" + f"<|Answer|>"
    examples["text"] = text
    examples["answer"] = answer
    examples["question"] = question
    return examples

def wsi_formatting_qa_open_test(examples, tokenizer):
    question = examples["question"]
    answer = examples["answer"]
    text = f"<|Question|>{question}" + f"<|Answer|>"
    examples["text"] = text
    examples["text_input"] = question
    return examples

def wsi_formatting_qa_close_test(examples, tokenizer, prompt_tag=True):
    question = examples["question"]
    answer = examples["answer"]
    if answer.lower() in ['yes', 'no']:
        prompt = f" Please provide only the answer (either Yes or No) for the following statement. Do not include any explanations or additional text. Just give Yes or No."
    else:
        prompt = f" Please provide only the answer (for example, A. [Answer Text], B. [Answer Text], etc.) for the following question. Do not include any explanations or additional text. Just give the letter followed by the corresponding answer."
    
    if prompt_tag:
        text = f"<|Question|>{question}<|Prompt|>{prompt}" + f"<|Answer|>"
    else:
        text = f"<|Question|>{question}" + f"<|Answer|>"

    examples["text"] = text
    examples["text_input"] = question
    return examples