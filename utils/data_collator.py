
from dataclasses import dataclass
from typing import Dict, List
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping
from transformers.data.data_collator import pad_without_fast_tokenizer_warning, _torch_collate_batch, _numpy_collate_batch
import numpy as np

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
class MyDataCollatorForLanguageModeling(DataCollatorMixin):
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
        
        temp_list = []
        for d in examples:
            # print(np.array(d["image"]).shape)
            temp_list.append(self.image_processor(d["image"]))
            del d["image"]

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
        batch["image"] = torch.stack(temp_list)

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
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        batch["image"] = torch.stack(img_list)
        batch["answers"] = ans_list

        return batch
