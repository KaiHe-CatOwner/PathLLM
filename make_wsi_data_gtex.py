import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, load_dataset

# set your file dir path
cache_dir = "/bask/projects/p/phwq4930-gbm/Zeyu/PathVLM/.cache"
npy_directory = '/bask/projects/p/phwq4930-gbm/Zeyu/WSI_Dataset/Conch/GTEx-Normal'
dataset_caption = 'CNX-PathLLM/Normal-Caption'
dataset = load_dataset(dataset_caption, split="train", cache_dir=cache_dir)
dataset = dataset.to_pandas()

df_text = dataset[['Tissue Sample ID', 'caption']]

feature1_files = glob.glob(os.path.join(npy_directory, "*_0_1024.npy"))
feature2_files = glob.glob(os.path.join(npy_directory, "*_1_512.npy"))
feature3_files = glob.glob(os.path.join(npy_directory, "*_1_1024.npy"))
# text_files = glob.glob(os.path.join(txt_directory, "*.txt"))
df_fea1 = pd.DataFrame(feature1_files, columns=['fea1_file_path'])
df_fea2 = pd.DataFrame(feature2_files, columns=['fea2_file_path'])
df_fea3 = pd.DataFrame(feature3_files, columns=['fea3_file_path'])
# df_text = pd.DataFrame(text_files, columns=['text_file_path'])
df_fea1['Tissue Sample ID'] = df_fea1['fea1_file_path'].apply(lambda x: os.path.basename(x).split('_')[0])
df_fea2['Tissue Sample ID'] = df_fea2['fea2_file_path'].apply(lambda x: os.path.basename(x).split('_')[0])
df_fea3['Tissue Sample ID'] = df_fea3['fea3_file_path'].apply(lambda x: os.path.basename(x).split('_')[0])

df_fea = pd.merge(df_fea1, df_fea2, on='Tissue Sample ID', how='inner')
df_fea = pd.merge(df_fea, df_fea3, on='Tissue Sample ID', how='inner')
df = pd.merge(df_fea, df_text, on='Tissue Sample ID', how='inner')

print(df.head())
print(df.shape)

data_index = 0
data = []
for i in tqdm(range(df.shape[0])):
    
    feature1_content = np.load(df.iloc[i]['fea1_file_path'], allow_pickle=True) 
    feature1 = feature1_content[()]['feature'].cpu().numpy().flatten() # Nx512
    feature1_cor = np.array([x.split('_')[:2] for x in feature1_content[()]['index']]).astype('int').flatten() # Nx2
    
    feature2_content = np.load(df.iloc[i]['fea2_file_path'], allow_pickle=True)
    feature2 = feature2_content[()]['feature'].cpu().numpy().flatten()
    feature2_cor = np.array([x.split('_')[:2] for x in feature2_content[()]['index']]).astype('int').flatten()
    
    feature3_content = np.load(df.iloc[i]['fea3_file_path'], allow_pickle=True)
    feature3 = feature3_content[()]['feature'].cpu().numpy().flatten()
    feature3_cor = np.array([x.split('_')[:2] for x in feature3_content[()]['index']]).astype('int').flatten()

    txt_content = df.iloc[i]['caption']
    tissue_id = df.iloc[i]['Tissue Sample ID']
        
    data.append({'f1': feature1, 'cor1': feature1_cor, 
                 'f2': feature2, 'cor2': feature2_cor, 
                 'f3': feature3, 'cor3': feature3_cor, 
                 'label': txt_content, 'id': tissue_id})
    
    if (i + 1) % 3000 == 0 or i == df.shape[0] - 1:
        data_df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(data_df)
        dataset.save_to_disk(f'/bask/projects/p/phwq4930-gbm/Zeyu/WSI_Dataset/GTExData_part{data_index}')
        data_index += 1
        data = []  # Reset data list for the next batch
