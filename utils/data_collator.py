import os
from dataclasses import dataclass, field
from typing import Dict, List
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping
from transformers.data.data_collator import pad_without_fast_tokenizer_warning, _torch_collate_batch
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

class DataCollatorMixin:
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "tf":
            return self.tf_call(features)
        elif return_tensors == "pt":
            return self.torch_call(features)
        elif return_tensors == "np":
            return self.numpy_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")
@dataclass
class MyDataCollatorForQFormerPatchPretrain(DataCollatorMixin):
    image_processor: Any
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = False
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __post_init__(self): 
        if self.mlm:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def _resize_image(self, image, min_size=448, max_size=1024):
        """
        Resize the image such that the shortest side is min_size while maintaining aspect ratio.
        """
        width, height = image.size
        if width < min_size or height < min_size:
            if width < height:
                new_width = min_size
                new_height = int(height * (min_size / width))
            else:
                new_height = min_size
                new_width = int(width * (min_size / height))
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        elif width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return image

    def _crop_image(self, image, crop_size=448, overlap=0.5):
        """
        Crop the image into patches of crop_size with a specified overlap.
        """
        width, height = image.size
        step = int(crop_size * (1 - overlap))
        patches = []
        for top in range(0, height - crop_size + 1, step):
            for left in range(0, width - crop_size + 1, step):
                box = (left, top, left + crop_size, top + crop_size)
                patch = image.crop(box)
                patches.append(patch)
        
        # Handling right and bottom edges
        if width % crop_size != 0:
            for top in range(0, height - crop_size + 1, step):
                box = (width - crop_size, top, width, top + crop_size)
                patch = image.crop(box)
                patches.append(patch)
        if height % crop_size != 0:
            for left in range(0, width - crop_size + 1, step):
                box = (left, height - crop_size, left + crop_size, height)
                patch = image.crop(box)
                patches.append(patch)
        if width % crop_size != 0 and height % crop_size != 0:
            box = (width - crop_size, height - crop_size, width, height)
            patch = image.crop(box)
            patches.append(patch)
        
        return patches

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        
        patch_list = []
        num_list = []
        text_list = []

        for d in examples:
            # print(np.array(d["image"]).shape)
            image = self._resize_image(d["image"])
            patches = self._crop_image(image) # [224 x 224]
            patches = [self.image_processor(patch) for patch in patches] # [448x448]
            patch_list += patches
            num_list.append(len(patches))
            del d["image"]

        for d in examples:
            text_list.append(d["text"])
            del d["text"]

        batch = {"text": text_list}
        batch["image"] = torch.stack(patch_list)
        batch["patch_num"] = num_list

        return batch

@dataclass
class MyDataCollatorForQFormerPatchInstruct(MyDataCollatorForQFormerPatchPretrain):
    tokenizer: PreTrainedTokenizerBase
    image_processor: Any
    mlm: bool = False
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"
    test: bool = False
        
    def pad_token_id_list(self, input_id_list, padding_value=0):
        """
        Pad the list of token ID lists to the maximum length of lists in the input.
        
        Args:
            input_id_list (List[List[int]]): List of token ID lists, each list represents a sequence of token IDs.
            padding_value (int, optional): The value used for padding shorter lists. Defaults to 0.
        
        Returns:
            List[List[int]]: A new list where all inner lists are padded to the maximum length found in the original list.
        """
        # Find the maximum length of the lists in the input
        max_length = max(len(inner_list) for inner_list in input_id_list)
        
        # Create a new list where each inner list is padded to the maximum length
        padded_list = [inner_list + [padding_value] * (max_length - len(inner_list)) for inner_list in input_id_list]
        
        return padded_list
    
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        
        patch_list = []
        num_list = []
        input_id_list = []
        text_list = []
        ans_list = []
        attention_mask_list = []
        text_input_list = []

        for d in examples:
            image = self._resize_image(d["image"])
            patches = self._crop_image(image) # [448x448]
            patches = [self.image_processor(patch) for patch in patches] # [448x448]
            patch_list += patches
            num_list.append(len(patches))
            del d["image"]

        for d in examples:
            input_id_list.append(d["input_ids"])
            attention_mask_list.append(d["attention_mask"])
            text_input_list.append(d["text_input"])
            if self.test:
                text_list.append(d["text"])
                ans_list.append(d["answer"])
                del d["answer"]
                del d["text"]
            del d["text_input"]
        
        input_id_list = self.pad_token_id_list(input_id_list, self.tokenizer.pad_token_id)
        attention_mask_list = self.pad_token_id_list(attention_mask_list, 0)

        batch = {"input_ids": torch.tensor(input_id_list)}
        batch["attention_mask"] = torch.tensor(attention_mask_list)
        
        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        # batch = {"text": text_list}
        if self.test:
            batch["text"] = text_list
            batch["answers"] = ans_list
            
        batch["text_input"] = text_input_list
        batch["labels"] = labels
        batch["image"] = torch.stack(patch_list)
        batch["patch_num"] = num_list
        return batch

@dataclass
class MyDataCollatorForPPathVLM(MyDataCollatorForQFormerPatchInstruct):
    tokenizer: PreTrainedTokenizerBase
    image_processor: Any
    mlm: bool = False
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"
    test: bool = False


    def __post_init__(self): 
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
        self.question_token_id = self.tokenizer.convert_tokens_to_ids('<|Question|>')
        self.answer_token_id = self.tokenizer.convert_tokens_to_ids('<|Answer|>')
        self.pad_token_id = self.tokenizer.pad_token_id

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        
        patch_list = []
        num_list = []
        input_id_list = []
        attention_mask_list = []

        if self.test:
            ans_list = []

        for d in examples:
            image = self._resize_image(d["image"])
            patches = self._crop_image(image) # [448x448]
            patches = [self.image_processor(patch) for patch in patches] # [448x448]
            patch_list += patches
            num_list.append(len(patches))
            del d["image"]

        for d in examples:
            if self.test:
                ans_list.append(d["answer"])
                del d["answer"]
            input_id_list.append(d["input_ids"])
            attention_mask_list.append(d["attention_mask"])
        
        input_id_list = self.pad_token_id_list(input_id_list, self.tokenizer.pad_token_id)
        attention_mask_list = self.pad_token_id_list(attention_mask_list, 0)

        batch = {"input_ids": torch.tensor(input_id_list)}
        batch["attention_mask"] = torch.tensor(attention_mask_list)
        
        labels = batch["input_ids"].clone()
        # process labels -> -100
        labels[labels == 128000] = -100
        labels[labels == self.pad_token_id] = -100

        for row in labels:
            # 处理 pad_token_id
            # positions = (row == self.pad_token_id).nonzero(as_tuple=True)[0]
            # if len(positions) > 1:
            #     row[positions[1:]] = -100  # 保留第一个 pad_token_id，其他设置为 -100

            # 处理 question_token_id 和 answer_token_id
            start_idx = (row == self.question_token_id).nonzero(as_tuple=True)[0]
            end_idx = (row == self.answer_token_id).nonzero(as_tuple=True)[0]

            # 确保 start_idx 和 end_idx 不为空且为单一整数
            if len(start_idx) > 0 and len(end_idx) > 0:
                start_idx = start_idx[0].item()  # 获取第一个匹配的索引并转换为整数
                end_idx = end_idx[0].item()      # 获取第一个匹配的索引并转换为整数

                if start_idx <= end_idx:
                    row[start_idx : end_idx + 1] = -100  # 将范围内的值设为 -100

        if self.test:
            batch["answers"] = ans_list
        batch["labels"] = labels
        batch["image"] = torch.stack(patch_list)
        batch["patch_num"] = num_list
        return batch

@dataclass
class MyDataCollatorForWPathVLM(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    fea_root: str = None
    agg_strategy: str = 'abmil'
    n_heads: List[int] = field(default_factory=lambda: [32, 16, 8])
    fea_name_list: List[str] = field(default_factory=lambda: ['f1024', 'f2048', 'f4096'])
    fea_dim: int = 512
    n_level: int = 3
    mlm: bool = False
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"
    test: bool = False

    def __post_init__(self): 
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
        self.question_token_id = self.tokenizer.convert_tokens_to_ids('<|Question|>')
        self.answer_token_id = self.tokenizer.convert_tokens_to_ids('<|Answer|>')
        self.pad_token_id = self.tokenizer.pad_token_id

    def __get_nic__(self, features, coords, size): 
        # NIC not use at this moment
        w = coords[:,0]
        h = coords[:,1]
        w_min = w.min()
        w_max = w.max()
        h_min = h.min()
        h_max = h.max()
        image_shape = [(w_max-w_min)//size+1,(h_max-h_min)//size+1]
        mask = np.ones((image_shape[0], image_shape[1]))
        features_nic = np.ones((features.shape[-1], image_shape[0], image_shape[1])) * np.nan
        coords_nic = -np.ones((image_shape[0], image_shape[1], 2))
        # Store each patch feature in the right position
        for patch_feature, x, y in zip(features, w, h):
            coord = [x,y]
            x_nic, y_nic = (x-w_min)//size, (y-h_min)//size
            features_nic[:, x_nic, y_nic] = patch_feature
            coords_nic[x_nic, y_nic] = coord
        # Populate NaNs
        mask[np.isnan(features_nic)[0]] = 0
        features_nic[np.isnan(features_nic)] = 0
        return features_nic, mask
    
    def __feature_trans__(self, examples: List[Union[List[int], Any, Dict[str, Any]]], key: str, cor: str):
        
        fea_list = []
        cor_list = []
        patch_masks = []
        max_dim = 0
        
        for d in examples:
            current_dim = d[key].shape[0]
            if current_dim > max_dim:
                max_dim = current_dim

        for d in examples:
            original_data = d[key]
            original_cor = d[cor]
            current_dim =  d[key].shape[0]

            padded_data = np.zeros([max_dim, self.fea_dim])
            cor_data = np.zeros([max_dim, 2], dtype=int)
            patch_mask = np.zeros(max_dim, dtype=int)

            padded_data[:current_dim, :] = original_data
            patch_mask[:int(current_dim)] = 1
            cor_data[:int(current_dim), :] = original_cor
            

            fea_list.append(torch.from_numpy(padded_data))
            cor_list.append(torch.from_numpy(cor_data))
            patch_masks.append(patch_mask)
        
        return fea_list, cor_list, patch_masks

    def __load_full_feature__(self, fea_path_ori: str):

        fea_path = '/'.join(fea_path_ori.split('/')[-2:])
        fea_path = os.path.join(self.fea_root, fea_path)
        fea = np.load(fea_path, allow_pickle=True)
        f = fea[()]['feature']
        cor = fea[()]['index']
        cor = np.array([filename.split('_')[:2] for filename in cor], dtype=int)

        return f, cor

    def __sample_feature__(self, f, cor, n_head: int):
        
        num_samples = cor.shape[0]

        # 如果n_head大于或等于cor的数量，保留全部数据
        if n_head >= num_samples:
            return f, cor

        # 使用 KMeans 将坐标分成 n_head 个组
        kmeans = KMeans(n_clusters=n_head, random_state=42)
        labels = kmeans.fit_predict(cor)  # 为每个坐标分配一个聚类标签

        sampled_indices = []

        # 遍历每个聚类标签，从每个组中随机选择一个点
        for i in range(n_head):
            group_indices = np.where(labels == i)[0]
            if len(group_indices) > 0:
                sampled_index = np.random.choice(group_indices, 1)[0]  # 随机选择一个样本
                sampled_indices.append(sampled_index)

        # 获取采样后的特征和坐标
        f_sampled = f[sampled_indices]
        cor_sampled = cor[sampled_indices]

        return f_sampled, cor_sampled

    def __load_clusters_feature__(self, fea_path_ori: str):

        fea_path = '/'.join(fea_path_ori.split('/')[-2:])
        fea_path = fea_path.split('_')[0]
        fea_path = fea_path.split('.')[0] + '.npy'
        fea_path = os.path.join(self.fea_root, fea_path)
        fea = np.load(fea_path, allow_pickle=True)
        f1 = fea[()]['f1024']
        f2 = fea[()]['f2048']
        f3 = fea[()]['f4096']
        cor1 = np.zeros((f1.shape[0], 2), dtype=int)
        cor2 = np.zeros((f2.shape[0], 2), dtype=int)
        cor3 = np.zeros((f3.shape[0], 2), dtype=int)

        return [f1, f2, f3], [cor1, cor2, cor3]

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:

        fea_list = []
        cor_list = []
        patch_mask_list = []
        exa_list = []
        instructs = []

        if self.test:
            ans_list = []
            qes_list = []
            slide_id_list = []
            for d in examples:
                qes_list.append(d["question"])
                ans_list.append(d["answer"])
                slide_id_list.append(d["slide_id"])
                del d["question"],d["answer"],d["slide_id"]
                
        # load img embeddings from local npy
        for d in examples:
            exa = {}

            for i in range(len(self.fea_name_list)):
                fea_name = self.fea_name_list[i]
                fea_path_ori = d[fea_name]
                del d[fea_name]
                if self.agg_strategy in ['abmil','longnet','qformer']:
                    f, cor = self.__load_full_feature__(fea_path_ori)
                elif self.agg_strategy == 'sample':
                    f, cor = self.__load_full_feature__(fea_path_ori)
                    f, cor = self.__sample_feature__(f, cor, self.n_heads[i])
                else:
                    continue
                exa['f{}'.format(i)] = f
                exa['cor{}'.format(i)] = cor
            
            if self.agg_strategy in ['kmeans','gmm']: # kmeans, gmm
                f, cor = self.__load_clusters_feature__(fea_path_ori)
                for i in range(len(f)):
                    exa['f{}'.format(i)] = f[i]
                    exa['cor{}'.format(i)] = cor[i]

            exa_list.append(exa)

        # transfer to list
        for level in range(self.n_level):
            fea, cor, patch_mask = self.__feature_trans__(exa_list, "f{}".format(level), "cor{}".format(level))
            fea_list.append(fea)
            cor_list.append(cor)
            patch_mask_list.append(patch_mask)

        # for instruct ids and mask padding
        if "input_ids_instruct" in examples[0].keys():
            for d in examples:
                instruct = {}
                instruct["input_ids"] = d["input_ids_instruct"]
                instruct["attention_mask"] = d["attention_mask_instruct"]
                instructs.append(instruct)
                del d["input_ids_instruct"],d["attention_mask_instruct"]
        
            instruct_batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, instructs, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
            )

        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        labels = batch["input_ids"].clone()

        # print(batch["input_ids"])
        # print(batch["attention_mask"])

        # print("####pre:")
        # print(labels)

        # process labels -> -100
        labels[labels == 128000] = -100
        labels[labels == self.pad_token_id] = -100

        for row in labels:
            # # 处理 pad_token_id
            # positions = (row == self.pad_token_id).nonzero(as_tuple=True)[0]
            # if len(positions) > 1:
            #     row[positions[1:]] = -100  # 保留第一个 pad_token_id，其他设置为 -100

            # 处理 question_token_id 和 answer_token_id
            start_idx = (row == self.question_token_id).nonzero(as_tuple=True)[0]
            end_idx = (row == self.answer_token_id).nonzero(as_tuple=True)[0]

            # 确保 start_idx 和 end_idx 不为空且为单一整数
            if len(start_idx) > 0 and len(end_idx) > 0:
                start_idx = start_idx[0].item()  # 获取第一个匹配的索引并转换为整数
                end_idx = end_idx[0].item()      # 获取第一个匹配的索引并转换为整数

                if start_idx <= end_idx:
                    row[start_idx : end_idx + 1] = -100  # 将范围内的值设为 -100

        # print("####post:")
        # print(labels)
        
        batch["labels"] = labels
        if instructs:
            batch["input_ids_instruct"] = instruct_batch["input_ids"][:,1:]
            batch["attention_mask_instruct"] = instruct_batch["attention_mask"][:,1:]

        if self.test:
            batch["answers"] = ans_list
            batch["questions"] = qes_list
            batch["slide_ids"] = slide_id_list

        for level in range(self.n_level):
            batch["fea{}".format(level)] = torch.stack(fea_list[level])
            batch["mask{}".format(level)] = torch.from_numpy(np.array(patch_mask_list[level], dtype=int))
            batch["cor{}".format(level)] = torch.stack(cor_list[level])

        return batch

# @dataclass
# class MyDataCollatorForWPathVLMTest(MyDataCollatorForWPathVLM):
#     tokenizer: PreTrainedTokenizerBase
#     fea_root: str = None
#     agg_strategy: str = 'abmil'
#     n_heads: List[int] = field(default_factory=lambda: [32, 16, 8])
#     fea_name_list: List[str] = field(default_factory=lambda: ['f1024', 'f2048', 'f4096'])
#     fea_dim: int = 512
#     n_level: int = 3
#     mlm: bool = False
#     mlm_probability: float = 0.15
#     pad_to_multiple_of: Optional[int] = None
#     tf_experimental_compile: bool = False
#     return_tensors: str = "pt"

#     def __post_init__(self): 
#         if self.mlm and self.tokenizer.mask_token is None:
#             raise ValueError(
#                 "This tokenizer does not have a mask token which is necessary for masked language modeling. "
#                 "You should pass `mlm=False` to train on causal language modeling instead."
#             )

#     def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:

#         fea_list = []
#         cor_list = []
#         patch_mask_list = []
#         ans_list = []

#         for d in examples:
#             ans_list.append(d["answer"])
#             del d["answer"]

#         for level in range(self.n_level):
#             fea, cor, patch_mask = self.__feature_trans__(examples, "f{}".format(level+1), "cor{}".format(level+1))
#             fea_list.append(fea)
#             cor_list.append(cor)
#             patch_mask_list.append(patch_mask)

#         if isinstance(examples[0], Mapping):
#             batch = pad_without_fast_tokenizer_warning(
#                 self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
#             )
#         else:
#             batch = {
#                 "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
#             }

#         # If special token mask has been preprocessed, pop it from the dict.
#         labels = batch["input_ids"].clone()
#         if self.tokenizer.pad_token_id is not None:
#             labels[labels == self.tokenizer.pad_token_id] = -100

#         batch["labels"] = labels
#         batch["answers"] = ans_list

#         for level in range(self.n_level):
#             batch["fea{}".format(level+1)] = torch.stack(fea_list[level])
#             batch["mask{}".format(level+1)] = torch.from_numpy(np.array(patch_mask_list[level], dtype=int))
#             batch["cor{}".format(level+1)] = torch.stack(cor_list[level])

#         return batch    

# @dataclass
# class MyDataCollatorForPPathVLMTest(MyDataCollatorForPPathVLM):
#     tokenizer: PreTrainedTokenizerBase
#     image_processor: Any
#     mlm: bool = False
#     mlm_probability: float = 0.15
#     pad_to_multiple_of: Optional[int] = None
#     tf_experimental_compile: bool = False
#     return_tensors: str = "pt"

#     def __post_init__(self): 
#         if self.mlm and self.tokenizer.mask_token is None:
#             raise ValueError(
#                 "This tokenizer does not have a mask token which is necessary for masked language modeling. "
#                 "You should pass `mlm=False` to train on causal language modeling instead."
#             )
        
#     def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        
#         patch_list = []
#         num_list = []
#         ans_list = []
#         input_id_list = []
#         attention_mask_list = []

#         for d in examples:
#             image = self._resize_image(d["image"])
#             patches = self._crop_image(image) # [448x448]
#             # print(patches)
#             patches = [self.image_processor(patch) for patch in patches] # [448x448]
#             # patches = self.image_processor(d["image"]) 
#             # print(patches)
#             patch_list += patches
#             num_list.append(len(patches))
#             del d["image"]

#         for d in examples:
#             ans_list.append(d["answer"])
#             input_id_list.append(d["input_ids"])
#             attention_mask_list.append(d["attention_mask"])
#             del d["answer"]
        
#         input_id_list = self.pad_token_id_list(input_id_list, self.tokenizer.pad_token_id)
#         attention_mask_list = self.pad_token_id_list(attention_mask_list, 0)

#         batch = {"input_ids": torch.tensor(input_id_list)}
#         batch["attention_mask"] = torch.tensor(attention_mask_list)
        
#         labels = batch["input_ids"].clone()
#         if self.tokenizer.pad_token_id is not None:
#             labels[labels == self.tokenizer.pad_token_id] = -100

#         # batch = {"text": text_list}
#         batch["labels"] = labels
#         batch["image"] = torch.stack(patch_list)
#         batch["patch_num"] = num_list
#         batch["answers"] = ans_list
#         return batch


# @dataclass
# class MyDataCollatorForLanguageModelingTest(DataCollatorMixin):
#     tokenizer: PreTrainedTokenizerBase
#     image_processor: Any
#     mlm: bool = False
#     mlm_probability: float = 0.15
#     pad_to_multiple_of: Optional[int] = None
#     tf_experimental_compile: bool = False
#     return_tensors: str = "pt"

#     def __post_init__(self): 
#         if self.mlm and self.tokenizer.mask_token is None:
#             raise ValueError(
#                 "This tokenizer does not have a mask token which is necessary for masked language modeling. "
#                 "You should pass `mlm=False` to train on causal language modeling instead."
#             )

#     def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        
#         img_list = []
#         ans_list = []
#         for d in examples:
#             # print(np.array(d["image"]).shape)
#             img_list.append(self.image_processor(d["image"]))
#             ans_list.append(d["answer"])
#             del d["image"]
#             del d["answer"]

#         if isinstance(examples[0], Mapping):
#             batch = pad_without_fast_tokenizer_warning(
#                 self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
#             )
#         else:
#             batch = {
#                 "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
#             }

#         # If special token mask has been preprocessed, pop it from the dict.
#         labels = batch["input_ids"].clone()

#         # do not apply loss to pad, ques
#         if self.tokenizer.pad_token_id is not None:
#             labels[labels == self.tokenizer.pad_token_id] = -100

#         batch["labels"] = labels
#         batch["image"] = torch.stack(img_list)
#         batch["answers"] = ans_list

#         return batch
