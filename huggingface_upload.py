import os
from datasets import load_from_disk, concatenate_datasets

# 函数：从磁盘加载并合并多个数据集
def load_and_merge_datasets(directories):
    datasets = []
    for directory in directories:
        # 从磁盘加载数据集
        dataset = load_from_disk(directory)
        datasets.append(dataset)
    
    # 合并所有数据集
    merged_dataset = concatenate_datasets(datasets)
    return merged_dataset

# 示例数据集目录列表
data_directories = ["/bask/projects/p/phwq4930-gbm/Zeyu/WSI_Dataset/WVLMdata_part0", 
                    "/bask/projects/p/phwq4930-gbm/Zeyu/WSI_Dataset/WVLMdata_part1",
                    "/bask/projects/p/phwq4930-gbm/Zeyu/WSI_Dataset/WVLMdata_part2", 
                    "/bask/projects/p/phwq4930-gbm/Zeyu/WSI_Dataset/WVLMdata_part3"]

# 读取并合并数据集
merged_dataset = load_and_merge_datasets(data_directories)

merged_dataset.push_to_hub("CNX-PathLLM/TCGA-WSI-Text")
