import pandas as pd
from datasets import load_dataset, DatasetDict

k = 10

# split_path = 'indices_and_slide_ids.csv'
# df = pd.read_csv(split_path)

# # 定义一个函数，对每个 project 分组并随机打乱，然后添加 fold 列
# def shuffle_and_add_fold(df_group):
#     # 随机打乱
#     df_group = df_group.sample(frac=1, random_state=42).reset_index(drop=True)
#     # 添加 fold 列，从 0 到 9 循环
#     df_group['fold'] = [i % k for i in range(len(df_group))]
#     return df_group

# # 根据 project 列分组，并应用上述函数
# df_shuffled = df.groupby('project', group_keys=False).apply(shuffle_and_add_fold)

# # 保存结果到新的 CSV 文件
# df_shuffled.to_csv('indices_and_slide_ids_with_folds.csv', index=False)

dataset = load_dataset('CNX-PathLLM/TCGA-WSI-Text', split='train', cache_dir='/bask/projects/p/phwq4930-gbm/Zeyu/PathVLM/.cache')

# 创建一个空的 DatasetDict
dataset_dict = DatasetDict()

df_indices = pd.read_csv('./dataset_csv/indices_and_slide_ids_with_folds.csv')

# 分割数据集并添加到 DatasetDict
for i in range(k):
    fold_indices = df_indices[df_indices['fold'] == i]['index'].tolist()
    fold_dataset = dataset.select(fold_indices)
    # 添加到 DatasetDict 中
    dataset_dict[f'fold_{i}'] = fold_dataset

print(dataset_dict)

dataset_dict.save_to_disk('/bask/projects/p/phwq4930-gbm/Zeyu/WSI_Dataset/TCGA-WSI-Text-Folds')

# dataset_dict.push_to_hub('CNX-PathLLM/TCGA-WSI-Text-Folds')