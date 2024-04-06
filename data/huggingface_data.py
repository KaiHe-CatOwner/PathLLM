import pandas as pd
import datasets
import re
import os
import shutil
#分割训练集、测试集、验证集
splits = ["test","train","val"]
#分数据集处理
for item in splits:
    os.makedirs(f"our_clean/{item}/", exist_ok=True)
    data = pd.read_csv(f"{item}.csv")
    data["image_path"] = data["image_path"].map(lambda x:x.split("/")[-1])
    #简单清洗文本内容
    f = lambda x: re.sub(' +', ' ', str(x).lower()).replace(" ?", "?").strip()
    #huggingface要求需要包含file_name作为键值
    data.insert(0, "file_name", "")
    data["question"] = data["question"].apply(f)
    data["answer"] = data["answer"].apply(f)
    #实现图文对应
    for i, row in data.iterrows():
        file_name = f"img_{i}.jpg"
        data["file_name"].iloc[i] = file_name
        shutil.copyfile(src=f"author-folder/pvqa/pvqa/images/{item}/{row['image']}.jpg", dst=f"our_clean/{item}/{file_name}")
    ##删除无关行
    _ = data.pop("image")
    data.drop(["pathology","image_path"],axis=1,inplace=True)
    data.to_csv(f"our_clean/{item}/metadata.csv", index=False)
#创建imagefolder格式的数据集，data_dir为存放数据的文件夹，可以参考https://huggingface.co/docs/datasets/en/image_load
dataset = datasets.load_dataset("imagefolder", data_dir="our_clean/")
#发布数据
dataset.push_to_hub("CNX-PathLLM/PVQAClean")