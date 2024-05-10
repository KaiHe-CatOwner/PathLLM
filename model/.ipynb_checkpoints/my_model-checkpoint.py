# -*- coding:utf-8 -*-
import os
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from accelerate import Accelerator
import open_clip
from utils.utils import clip_path_map


class MyCustomModel(nn.Module):
    def __init__(self, llm_requires_grad, clip_name, load_in_8bit, load_in_4bit, llm_name, trust_remote_code, token, tokenizer, image_token_id):
        nn.Module.__init__(self)

        self.clip_name = clip_name

        self.vision_encoder, self.image_processor, self.embed_dim = self.load_vision_encoder()

        self.llm_tokenizer = tokenizer
        self.llm = self.load_llm(load_in_8bit, load_in_4bit, llm_name, trust_remote_code, token)
        self.embedding_layer = self.llm.get_input_embeddings()

        self.fusion_layer_S = nn.Linear(self.embed_dim, self.llm.config.hidden_size)
        self.fusion_layer_E = torch.nn.TransformerEncoderLayer(self.llm.config.hidden_size, 8,  batch_first=True)
        
        self.config = self.llm.config
        self.image_token_id = image_token_id
        self.vision_encoder.requires_grad = False
        self.llm.requires_grad = llm_requires_grad


    def load_vision_encoder(self):
        print("vision_encoder loading ...")

        clip_path = clip_path_map(self.clip_name)
        if self.clip_name=="pathclip-base":
            vision_encoder, _, image_processor = open_clip.create_model_and_transforms('ViT-B-16', pretrained=clip_path, force_quick_gelu=True)
            embed_dim = 512
            vision_encoder.visual.output_tokens = True
        elif self.clip_name=="conch": 
            from conch.open_clip_custom import create_model_from_pretrained
            vision_encoder, image_processor = create_model_from_pretrained('conch_ViT-B-16', clip_path)
            embed_dim = 512
            vision_encoder.visual.output_tokens = True
        elif self.clip_name=="uni": 
            vision_encoder = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
            vision_encoder.load_state_dict(torch.load(clip_path, map_location="cpu"), strict=True)
            def rgba_to_rgb(image):
                return image.convert('RGB')
            image_processor = transforms.Compose(
                                [
                                    transforms.Lambda(rgba_to_rgb),
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                ]
                            )
            embed_dim = 1024
        else:
            raise Exception("wrong clip")
        return vision_encoder, image_processor, embed_dim
    
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
            input_ids = kwargs["input_ids"]
            image = kwargs["image"]
            attention_mask = kwargs["attention_mask"]
            fusion_embs = self.get_fusion_embedding(input_ids, image)
            attention_mask = self.pad_attention_fusion(fusion_embs.size(1), attention_mask)
            res = self.llm.generate(inputs_embeds=fusion_embs, attention_mask=attention_mask, generation_config=generation_config)

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
                    cache_dir = "/bask/projects/p/phwq4930-gbm/Zeyu/PathVLM/.cache"
                )
        llm.resize_token_embeddings(len(self.llm_tokenizer))
        return llm
    
    def pad_attention_fusion(self, new_seq_len, need_pad_seq):
        padd_len = new_seq_len - need_pad_seq.size(1)
        bz = need_pad_seq.size(0)

        generated_pad = torch.ones((bz, padd_len), dtype=need_pad_seq.dtype).to(need_pad_seq.device)

        if self.llm_tokenizer.padding_side == "right":
            paded_seq = torch.cat((need_pad_seq, generated_pad), dim=1)
        else:
            paded_seq = torch.cat((generated_pad, need_pad_seq), dim=1)

        return paded_seq
    
    def pad_label_fusion(self, new_seq_len, labels):
        padd_len = new_seq_len - labels.size(1)
        bz = labels.size(0)

        generated_pad = torch.ones((bz, padd_len), dtype=labels.dtype).fill_(-100).to(labels.device)
        paded_seq = torch.cat((generated_pad, labels), dim=1)

        return paded_seq


    def get_fusion_embedding(self, input_ids, image):
        token_embs = self.embedding_layer(input_ids)
        if self.clip_name == 'uni':
            image_embs = self.vision_encoder(image)
        elif self.clip_name == 'conch'::
            image_embs = self.vision_encoder.encode_image(image, normalize=False, proj_contrast=False)
        else: 
            image_embs = self.vision_encoder.encode_image(image, normalize=False) # no proj_contrast=False for clip
        # print(image_embs.shape)
        # image_embs = torch.stack(image_embs, dim=0)
        if self.clip_name == 'pathclip-base':
            mapped_image_embs = self.fusion_layer_S(image_embs[0]) # shape (bz, 512) - (bz, 4096)
        else:
            mapped_image_embs = self.fusion_layer_S(image_embs) # shape (bz, 512) - (bz, 4096)
        mapped_image_embs = self.fusion_layer_E(mapped_image_embs) # shape (bz, 4096)
        mapped_image_embs = mapped_image_embs.unsqueeze(1) # shape (bz, 1, llm_hidden_state)
        
        image_token_emb = self.embedding_layer(torch.tensor(self.image_token_id).to(mapped_image_embs.device))
        batch_image_token_emb = image_token_emb.repeat(mapped_image_embs.size(0), 1, 1)
        fusion_embs =  torch.cat((batch_image_token_emb, mapped_image_embs, token_embs), dim=1)
        return fusion_embs
    
    def forward(self, *args, **kwargs):
        input_ids = kwargs["input_ids"]
        image = kwargs["image"]
        attention_mask = kwargs["attention_mask"]
        labels = kwargs["labels"]
        fusion_embs = self.get_fusion_embedding(input_ids, image)
        attention_mask = self.pad_attention_fusion(fusion_embs.size(1), attention_mask)
        labels = self.pad_label_fusion(fusion_embs.size(1), labels)

        output = self.llm(inputs_embeds=fusion_embs, attention_mask=attention_mask, labels=labels)
        return output