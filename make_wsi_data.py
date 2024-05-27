import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

# set your file dir path
npy_directory = '/bask/projects/p/phwq4930-gbm/Zeyu/WSI_Dataset/Conch'
txt_directory = '/bask/projects/p/phwq4930-gbm/Zeyu/WSI_Dataset/TCGA_Report/TXT-GPT4o'

feature1_files = glob.glob(os.path.join(npy_directory, "*/*_0_1024.npy"))
feature2_files = glob.glob(os.path.join(npy_directory, "*/*_1_512.npy"))
feature3_files = glob.glob(os.path.join(npy_directory, "*/*_1_1024.npy"))
text_files = glob.glob(os.path.join(txt_directory, "*.txt"))
df_fea1 = pd.DataFrame(feature1_files, columns=['fea1_file_path'])
df_fea2 = pd.DataFrame(feature2_files, columns=['fea2_file_path'])
df_fea3 = pd.DataFrame(feature3_files, columns=['fea3_file_path'])
df_text = pd.DataFrame(text_files, columns=['text_file_path'])
df_fea1['slide_id'] = df_fea1['fea1_file_path'].apply(lambda x: os.path.basename(x).split('.')[0])
df_fea2['slide_id'] = df_fea2['fea2_file_path'].apply(lambda x: os.path.basename(x).split('.')[0])
df_fea3['slide_id'] = df_fea3['fea3_file_path'].apply(lambda x: os.path.basename(x).split('.')[0])

df_fea1['slide_id'] = df_fea1['fea1_file_path'].apply(lambda x: os.path.basename(x).split('.')[0])
df_fea2['slide_id'] = df_fea2['fea2_file_path'].apply(lambda x: os.path.basename(x).split('.')[0])
df_fea3['slide_id'] = df_fea3['fea3_file_path'].apply(lambda x: os.path.basename(x).split('.')[0])

df_fea = pd.merge(df_fea1, df_fea2, on='slide_id', how='inner')
df_fea = pd.merge(df_fea, df_fea3, on='slide_id', how='inner')

df_fea['patient_id'] = df_fea['slide_id'].apply(lambda x: x[:12])

df_text['patient_id'] = df_text['text_file_path'].apply(lambda x: os.path.basename(x).split('.')[0])

df = pd.merge(df_fea, df_text, on='patient_id', how='inner')

df['project'] = df['fea1_file_path'].apply(lambda x: x.split('/')[-2])

print(df.head())
print(df.shape)

data = []
for i in tqdm(range(0,3000)):
    feature1_content = np.load(df.iloc[i]['fea1_file_path'], allow_pickle=True) 
    feature1 = feature1_content[()]['feature'].cpu().numpy().flatten() # Nx512
    feature1_cor = np.array([x.split('_')[:2] for x in feature1_content[()]['index']]).astype('int').flatten() # Nx2
    
    feature2_content = np.load(df.iloc[i]['fea2_file_path'], allow_pickle=True)
    feature2 = feature2_content[()]['feature'].cpu().numpy().flatten()
    feature2_cor = np.array([x.split('_')[:2] for x in feature2_content[()]['index']]).astype('int').flatten()
    
    feature3_content = np.load(df.iloc[i]['fea3_file_path'], allow_pickle=True)
    feature3 = feature3_content[()]['feature'].cpu().numpy().flatten()
    feature3_cor = np.array([x.split('_')[:2] for x in feature3_content[()]['index']]).astype('int').flatten()
    with open(df.iloc[i]['text_file_path'], 'r') as txt_file:
        txt_content = txt_file.read()
        
    data.append({'f1': feature1, 'cor1': feature1_cor, 
                 'f2': feature2, 'cor2': feature2_cor, 
                 'f3': feature3, 'cor3': feature3_cor, 
                 'label': txt_content, 'slide_id': df['slide_id'].iloc[i],
                 'project': df['project'].iloc[i],
                })
    
data = pd.DataFrame(data)
dataset = Dataset.from_pandas(data)

# 保存数据集
#dataset.push_to_hub('CNX-PathLLM/TCGA-WSI-Text')
dataset.save_to_disk('/bask/projects/p/phwq4930-gbm/Zeyu/WSI_Dataset/WVLMdata_part0')
