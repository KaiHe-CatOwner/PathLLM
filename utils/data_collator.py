from dataclasses import dataclass
from typing import Dict, List
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping
from transformers.data.data_collator import pad_without_fast_tokenizer_warning, _torch_collate_batch
import numpy as np
from PIL import Image

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
                del d["text"]
            del d["text_input"]
        # if isinstance(examples[0], Mapping):
        #     batch = pad_without_fast_tokenizer_warning(
        #         self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
        #     )
        #     print("isinstance over!")
        # else:
        #     batch = {
        #         "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        #     }
        # If special token mask has been preprocessed, pop it from the dict.
        
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

    def __post_init__(self): 
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        
        patch_list = []
        num_list = []
        input_id_list = []
        attention_mask_list = []

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
        
        input_id_list = self.pad_token_id_list(input_id_list, self.tokenizer.pad_token_id)
        attention_mask_list = self.pad_token_id_list(attention_mask_list, 0)

        batch = {"input_ids": torch.tensor(input_id_list)}
        batch["attention_mask"] = torch.tensor(attention_mask_list)
        
        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        # batch = {"text": text_list}
        batch["labels"] = labels
        batch["image"] = torch.stack(patch_list)
        batch["patch_num"] = num_list
        return batch
    
@dataclass
class MyDataCollatorForPPathVLMTest(MyDataCollatorForPPathVLM):
    tokenizer: PreTrainedTokenizerBase
    image_processor: Any
    mlm: bool = False
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __post_init__(self): 
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
        
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        
        patch_list = []
        num_list = []
        ans_list = []
        input_id_list = []
        attention_mask_list = []

        for d in examples:
            image = self._resize_image(d["image"])
            patches = self._crop_image(image) # [448x448]
            # print(patches)
            patches = [self.image_processor(patch) for patch in patches] # [448x448]
            # patches = self.image_processor(d["image"]) 
            # print(patches)
            patch_list += patches
            num_list.append(len(patches))
            del d["image"]

        for d in examples:
            ans_list.append(d["answer"])
            input_id_list.append(d["input_ids"])
            attention_mask_list.append(d["attention_mask"])
            del d["answer"]
        
        input_id_list = self.pad_token_id_list(input_id_list, self.tokenizer.pad_token_id)
        attention_mask_list = self.pad_token_id_list(attention_mask_list, 0)

        batch = {"input_ids": torch.tensor(input_id_list)}
        batch["attention_mask"] = torch.tensor(attention_mask_list)
        
        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        # batch = {"text": text_list}
        batch["labels"] = labels
        batch["image"] = torch.stack(patch_list)
        batch["patch_num"] = num_list
        batch["answers"] = ans_list
        return batch

@dataclass
class MyDataCollatorForWPathVLM(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    fea_dim: int = 512
    n_level: int = 3
    mlm: bool = False
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __post_init__(self): 
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

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
            current_dim = len(d[key])
            if current_dim > max_dim:
                max_dim = current_dim

        for d in examples:
            original_data = d[key]
            original_cor = d[cor]
            current_dim = len(original_data)

            padded_data = np.zeros(max_dim)
            cor_data = np.zeros((int(max_dim/self.fea_dim)*2), dtype=int)
            patch_mask = np.zeros(int(max_dim/self.fea_dim), dtype=int)

            padded_data[:current_dim] = original_data
            patch_mask[:int(current_dim/self.fea_dim)] = 1
            cor_data[:int(current_dim/self.fea_dim)*2] = original_cor
            

            fea_list.append(torch.from_numpy(padded_data.reshape(int(max_dim/self.fea_dim), self.fea_dim)).float())
            cor_list.append(torch.from_numpy(cor_data.reshape(int(max_dim/self.fea_dim), 2)))
            patch_masks.append(patch_mask)
        
            del d[key], d[cor]
        
        return fea_list, cor_list, patch_masks
            
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:

        fea_list = []
        cor_list = []
        patch_mask_list = []
        # ans_list = []

        # for d in examples:
        #     ans_list.append(d["text"])
        #     del d["text"]

        for level in range(self.n_level):
            fea, cor, patch_mask = self.__feature_trans__(examples, "f{}".format(level+1), "cor{}".format(level+1))
            fea_list.append(fea)
            cor_list.append(cor)
            patch_mask_list.append(patch_mask)

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
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        # batch["answers"] = ans_list

        for level in range(self.n_level):
            batch["fea{}".format(level+1)] = torch.stack(fea_list[level])
            batch["mask{}".format(level+1)] = torch.from_numpy(np.array(patch_mask_list[level], dtype=int))
            batch["cor{}".format(level+1)] = torch.stack(cor_list[level])

        return batch

@dataclass
class MyDataCollatorForWPathVLMTest(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    fea_dim: int = 512
    n_level: int = 3
    mlm: bool = False
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __post_init__(self): 
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

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
            current_dim = len(d[key])
            if current_dim > max_dim:
                max_dim = current_dim

        for d in examples:
            original_data = d[key]
            original_cor = d[cor]
            current_dim = len(original_data)

            padded_data = np.zeros(max_dim)
            cor_data = np.zeros((int(max_dim/self.fea_dim)*2), dtype=int)
            patch_mask = np.zeros(int(max_dim/self.fea_dim), dtype=int)

            padded_data[:current_dim] = original_data
            patch_mask[:int(current_dim/self.fea_dim)] = 1
            cor_data[:int(current_dim/self.fea_dim)*2] = original_cor
            

            fea_list.append(torch.from_numpy(padded_data.reshape(int(max_dim/self.fea_dim), self.fea_dim)).float())
            cor_list.append(torch.from_numpy(cor_data.reshape(int(max_dim/self.fea_dim), 2)))
            patch_masks.append(patch_mask)
        
            del d[key], d[cor]
        
        return fea_list, cor_list, patch_masks
            
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:

        fea_list = []
        cor_list = []
        patch_mask_list = []
        ans_list = []

        for d in examples:
            ans_list.append(d["answer"])
            del d["answer"]

        for level in range(self.n_level):
            fea, cor, patch_mask = self.__feature_trans__(examples, "f{}".format(level+1), "cor{}".format(level+1))
            fea_list.append(fea)
            cor_list.append(cor)
            patch_mask_list.append(patch_mask)

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
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        batch["answers"] = ans_list

        for level in range(self.n_level):
            batch["fea{}".format(level+1)] = torch.stack(fea_list[level])
            batch["mask{}".format(level+1)] = torch.from_numpy(np.array(patch_mask_list[level], dtype=int))
            batch["cor{}".format(level+1)] = torch.stack(cor_list[level])

        return batch    

@dataclass
class MyDataCollatorForLanguageModelingTest(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    image_processor: Any
    mlm: bool = False
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __post_init__(self): 
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        
        img_list = []
        ans_list = []
        for d in examples:
            # print(np.array(d["image"]).shape)
            img_list.append(self.image_processor(d["image"]))
            ans_list.append(d["answer"])
            del d["image"]
            del d["answer"]

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

        # do not apply loss to pad, ques
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        batch["image"] = torch.stack(img_list)
        batch["answers"] = ans_list

        return batch
