from datasets import load_dataset, DatasetDict
from PIL import Image, ImageFile, UnidentifiedImageError
import io
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 假设您的数据集是通过 load_dataset 加载的
cache_dir = "/bask/projects/p/phwq4930-gbm/Zeyu/PathVLM/.cache"
dataset_name = "CNX-PathLLM/Pathcap"
dataset = load_dataset(dataset_name, split="train", cache_dir=cache_dir)

print(f"original dataset size: {len(dataset)}")

# 保存有效样本的索引
valid_indices = []

# 遍历数据集并检查每个样本
for idx in tqdm(range(len(dataset))):
    try:
        example = dataset[idx]
        # img = example["jpg"]
        # img.verify()  # 验证图像文件
        # valid_indices.append(idx)  # 如果图像有效，保存索引
        # 检查文本字段是否为字符串
        text = example["txt"]
        if not isinstance(text, str):
            raise ValueError(f"文本字段不是字符串: {text}")
        valid_indices.append(idx)
    except Exception as e:
        print(f"无法识别文件 {idx}: {e}")

# 根据有效样本的索引选择有效样本
filtered_dataset = dataset.select(valid_indices)

# 过滤掉无法加载的图像
# filtered_dataset = dataset.filter(lambda example: example["is_valid"])

# 打印过滤后的数据集大小
print(f"filtered dataset size: {len(filtered_dataset)}")

if len(dataset) != len(filtered_dataset):
    # 将过滤后的数据集转换为 DatasetDict
    filtered_dataset_dict = DatasetDict({"train": filtered_dataset})
    # 推送到原有的数据集地址
    filtered_dataset_dict.push_to_hub(dataset_name)
