# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from accelerate import Accelerator
import open_clip
from utils.utils import clip_path_map


class MyCustomModel(nn.Module):
    def __init__(self, clip_name, load_in_8bit, load_in_4bit, llm_name, trust_remote_code, token, tokenizer):
        nn.Module.__init__(self)

        self.vision_encoder, self.image_processor = self.load_vision_encoder(clip_name)

        self.llm_tokenizer = tokenizer
        self.llm = self.load_llm(load_in_8bit, load_in_4bit, llm_name, trust_remote_code, token)
        self.embedding_layer = self.llm.get_input_embeddings()

        self.fusion_layer_S = nn.Linear(self.vision_encoder.embed_dim, self.llm.config.hidden_size)
        self.fusion_layer_E = torch.nn.TransformerEncoderLayer(self.llm.config.hidden_size, 8,  batch_first=True)
        
        self.config = self.llm.config
        
        self.vision_encoder.requires_grad = False
        # self.llm.requires_grad = False


    def load_vision_encoder(self, clip_name):
        print("vision_encoder loading ...")

        clip_path = clip_path_map(clip_name)
        if clip_name=="pathclip-base":
            vision_encoder, _, image_processor = open_clip.create_model_and_transforms('ViT-B-16', pretrained=clip_path, force_quick_gelu=True)
        elif clip_name=="conch": 
            from conch.open_clip_custom import create_model_from_pretrained
            vision_encoder, image_processor = create_model_from_pretrained('conch_ViT-B-16', clip_path)
        else:
            raise Exception("wrong clip")
        vision_encoder.visual.output_tokens = True
        return vision_encoder, image_processor
    
    def generate(self, *args, **kwargs):
        generation_config = GenerationConfig(
                max_length=100,
                temperature=1.0,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.llm_tokenizer.eos_token_id,
                bos_token_id=self.llm_tokenizer.bos_token_id,
            )
        
        with torch.no_grad():
            fusion_embs = self.get_fusion_embedding(*args, **kwargs)
            res = self.llm.generate(inputs_embeds=fusion_embs, generation_config=generation_config)

        generate_list = []
        for item in res:
            generation = self.llm_tokenizer.decode(item, skip_special_tokens=True)
            generate_list.append(generation)
        return generate_list

    def load_llm(self, load_in_8bit, load_in_4bit, llm_name, trust_remote_code, token):
        print("llm loading ...")
        if load_in_8bit and load_in_4bit:
            raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
        elif load_in_8bit or load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit
            )
            # Copy the model to each device
            device_map = {"": Accelerator().local_process_index}
            torch_dtype = torch.bfloat16
        else:
            device_map = None
            quantization_config = None
            torch_dtype = None


        llm = AutoModelForCausalLM.from_pretrained(
                    llm_name,
                    quantization_config=quantization_config,
                    device_map=device_map,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch_dtype,
                    token=token,
                    use_cache= True,
                )
        
        return llm
    
    def get_fusion_embedding(self, *args, **kwargs):
        input_ids = kwargs["input_ids"]
        image = kwargs["image"]

        token_embs = self.embedding_layer(input_ids)
        image_embs = self.vision_encoder.encode_image(image, proj_contrast=False, normalize=False)
        mapped_image_embs = self.fusion_layer_S(image_embs) # shape (bz, 512) - (bz, 4096)
        mapped_image_embs = self.fusion_layer_E(mapped_image_embs) # shape (bz, 4096)
        mapped_image_embs = mapped_image_embs.unsqueeze(1).repeat(1, token_embs.size(1), 1)  # shape (bz, seq_len, llm_hidden_state)
        fusion_embs = token_embs + mapped_image_embs

        # fusion_embs = torch.cat((mapped_image_embs, token_embs), dim =1)
        # if self.llm_tokenizer.padding_side == "right":
        #     attention_mask = torch.cat((torch.ones((attention_mask.size(0), 1)).to(attention_mask.device), attention_mask), dim=1)
        # else:
        #     attention_mask = torch.cat((attention_mask, torch.ones((attention_mask.size(0), 1)).to(attention_mask.device)), dim=1)
        return fusion_embs
    
    def forward(self, *args, **kwargs):
        labels = kwargs["labels"]
        attention_mask = kwargs["attention_mask"]
        fusion_embs = self.get_fusion_embedding(*args, **kwargs)
        output = self.llm(inputs_embeds=fusion_embs, attention_mask=attention_mask, labels=labels)
        return output