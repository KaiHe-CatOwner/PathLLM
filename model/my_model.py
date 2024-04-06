# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoTokenizer
from accelerate import Accelerator
import open_clip
from utils.utils import clip_path_map



class MyCustomModel(nn.Module):
    def __init__(self, script_args, tokenizer):
        nn.Module.__init__(self)
        self.script_args = script_args
        self.llm_tokenizer = tokenizer
        self.load_vision_encoder(script_args)
        self.load_llm(script_args)
        self.fusion_layer = nn.Linear(self.vision_encoder.embed_dim, self.llm.config.hidden_size)
        self.config = self.llm.config

    def load_vision_encoder(self, script_args):
        print("vision_encoder loading ...")

        clip_name = clip_path_map(script_args.clip_name)
        if script_args.clip_name=="pathclip-base":
            self.vision_encoder, _, self.image_processor = open_clip.create_model_and_transforms('ViT-B-16', pretrained=clip_name, force_quick_gelu=True)
        elif script_args.clip_name=="conch": 
            from conch.open_clip_custom import create_model_from_pretrained
            self.vision_encoder, self.image_processor = create_model_from_pretrained('conch_ViT-B-16', clip_name)
        self.vision_encoder.visual.output_tokens = True
    
    def load_llm(self, script_args):
        print("llm loading ...")
        if script_args.load_in_8bit and script_args.load_in_4bit:
            raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
        elif script_args.load_in_8bit or script_args.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
            )
            # Copy the model to each device
            device_map = {"": Accelerator().local_process_index}
            torch_dtype = torch.bfloat16
        else:
            device_map = None
            quantization_config = None
            torch_dtype = None

        self.llm = AutoModelForCausalLM.from_pretrained(
                    script_args.llm_name,
                    quantization_config=quantization_config,
                    device_map=device_map,
                    trust_remote_code=script_args.trust_remote_code,
                    torch_dtype=torch_dtype,
                    token=script_args.token,
                )
        self.embedding_layer = self.llm.get_input_embeddings()
    
    def forward(self, *args, **kwargs):
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]
        labels = kwargs["labels"]
        image = kwargs["image"]

        token_embs = self.embedding_layer(input_ids)
        image_embs = self.vision_encoder.encode_image(image, proj_contrast=False, normalize=False)
        mapped_image_embs = self.fusion_layer(image_embs) # shape (bz, 512)
        mapped_image_embs = mapped_image_embs.unsqueeze(1).repeat(1, token_embs.size(1), 1)  # shape (bz, seq_len, llm_hidden_state)
        fusion_embs = token_embs + mapped_image_embs

        # fusion_embs = torch.cat((mapped_image_embs, token_embs), dim =1)
        # if self.llm_tokenizer.padding_side == "right":
        #     attention_mask = torch.cat((torch.ones((attention_mask.size(0), 1)).to(attention_mask.device), attention_mask), dim=1)
        # else:
        #     attention_mask = torch.cat((attention_mask, torch.ones((attention_mask.size(0), 1)).to(attention_mask.device)), dim=1)

        output = self.llm(inputs_embeds=fusion_embs, attention_mask=attention_mask, labels=labels)
        return output