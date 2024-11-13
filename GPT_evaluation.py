import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import httpx
import backoff
import argparse
import openai
from openai import OpenAI
import json

# @backoff.on_exception(backoff.expo,(SyntaxError,openai.APIError,openai.error.APIConnectionError,TimeoutError,KeyError,NameError,ValueError),max_time=10)
# def generate(prompt):
#     message = [
#                 # {"role": "system", "content": "You are a senior pathologist."},
#                 {"role": "user", "content": prompt},
#         ]
#     global total_length
#     total_length+=len(str(message))
#     return openai.ChatCompletion.create(
#         temperature=0.2,
#         model = "gpt-4o-mini",
#         response_format={"type": "json_object"},
#         messages=message,
#         timeout=httpx.Timeout(15.0, read=15.0, write=15.0, connect=20.0)
#     )

def generate(prompt):
    client = OpenAI(api_key=api_key)
    message = [
        {"role": "user", "content": prompt},
    ]
    global total_length
    total_length += len(str(message))

    try:
        chat_completion = client.chat.completions.create(
            messages=message,
            model="gpt-4o",
            # temperature=0.2
        )
        return chat_completion
    except Exception as e:
        print(f"Error generating completion: {e}")
        return None

def open_question_evaluate(content):
    quilt_prompt = f"""You will be provided with a question, a ground truth answer, and a model-generated result related to a pathology case. Your task is to evaluate the correctness of the model's predicted answer.
    Instructions for evaluation:
    1.	If the model's answer and the ground truth contain similar or equivalent information, the model's response is considered correct, even if the wording or structure differs.
    2.	If the model's response is broader but includes the correct answer or its key elements, it should still be considered correct.
    3.	If part of the model's answer matches the key information from the ground truth, the result is partially correct.
    4.	If there is no conceptual overlap between the model's response and the ground truth, then it should be considered incorrect.
    5.	For yes/no questions, focus only on the correctness of the yes/no response itself.
    In the response, provide only a number:
    -	1 if the model's result is correct,
    -	2 if it is partially correct,
    -	0 if it is incorrect.
    Ensure your assessment reflects the clinical context and evaluates the accuracy of the provided content. Here is the content: {content}
    """
    response = generate(quilt_prompt)
    # print(response)
    res = eval(response.choices[0].message.content.strip())
    # print(res)
    return res

def description_evaluation(content):
    quilt_prompt = f"""You are a senior pathologist. You will be provided with a question, an answer, and a model-generated description related to a pathology case. In the provided content, the key named answers represents ground truth. Your task is to evaluate the generated description based on the following criteria:
                    1. **Helpfulness**: How useful is the description in addressing the question?
                    2. **Relevance**: Does the description relate directly to the question and the pathology case?
                    3. **Accuracy**: Is the information presented in the description factually correct and aligned with the known pathology?
                    4. **Level of Detail**: Does the description provide sufficient detail to be informative without being overly verbose?
                    Please rate the generated description on a scale of 1 to 10 for each criterion, where 1 indicates poor performance and 10 indicates excellent performance. 
                    The final output should be only a float number range from 1 to 10, which represents the overall score based on your evaluation of all aspects. Consider the clinical context and ensure your assessment is objective and free from bias.
                    Here is the content: {content}
                    """
    response = generate(quilt_prompt)
    res = eval(response.choices[0].message.content.strip())
    return res


api_key = os.getenv("OPENAI_API_KEY")
data_name = "ckpt10500_open.csv"

parser = argparse.ArgumentParser(description="处理数据文件名")
parser.add_argument('--type', type=str, default='open', help='desc:处理描述；open:处理开放式问题')
parser.add_argument('--filename', type=str, help='数据文件名')

args = parser.parse_args()


if args.filename:
    data_name = args.filename
print(data_name)
open_file = pd.read_csv(data_name)
total_length= 0

if args.type=="open":
    open_file["LLM_result"] = None
    correct,partial,wrong=0,0,0
    step = 0
    start = 0
    res = ""
    

    for i in tqdm(range(start,len(open_file))):
        content = str(open_file.iloc[i][["questions","answers","results"]].to_dict())
        try:
            res = open_question_evaluate(content)
            if int(res)==1:
                correct+=1
                open_file.loc[i, "LLM_result"]=1
            elif int(res)==2:
                partial+=1
                open_file.loc[i, "LLM_result"]=0
            elif int(res)==0:
                wrong+=1
                open_file.loc[i, "LLM_result"]=-1
            else:
                print(res)
        except Exception as e:
            print("bee:", e)
            continue

    print("correct: ",correct," partial: ",partial," wrong: ",wrong)

    print("accuracy:",(correct+partial)/(correct+partial+wrong))
    name_root = ".".join(data_name.split(".")[:-1])
    open_file.to_csv(f"{name_root}_open_test_accuracy.csv")


elif args.type=="desc":
    open_file["LLM_score"] = None
    step = 0
    start = 0
    score_list = []
    for i in tqdm(range(start,len(open_file))):
        content = str(open_file.iloc[i][["questions","answers","results"]].to_dict())
        try:
            res = description_evaluation(content)
            open_file.loc[i, "LLM_score"]=res
            score_list.append(res)
        except:
            print("bee:",res)
            continue
    print("Average score:",sum(score_list)/(10*len(score_list)))
    name_root = ".".join(data_name.split(".")[:-1])
    open_file.to_csv(f"{name_root}_OMINI_open_LLM_score.csv")

else:
    raise NotImplementedError("Not implemneted!!!")

print("total cost:",total_length/1e6 * 0.15 *7.1)
