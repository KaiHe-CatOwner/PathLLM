import logging

import torch
import timm
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from torchvision import transforms
from transformers import BertTokenizer
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures

from accelerate import Accelerator
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig

from utils.utils import clip_path_map


def init_tokenizer(truncation_side="right", pretrain_name="bert-base-uncased"):
        tokenizer = BertTokenizer.from_pretrained(pretrain_name, truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer
    
def init_Qformer(num_query_token, vision_width, cross_attention_freq=2, pretrain_name="bert-base-uncased"):
    encoder_config = BertConfig.from_pretrained(pretrain_name)
    encoder_config.encoder_width = vision_width
    # insert cross-attention layer every other block
    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = cross_attention_freq
    encoder_config.query_length = num_query_token
    Qformer = BertLMHeadModel.from_pretrained(
        pretrain_name, config=encoder_config
    )
    query_tokens = nn.Parameter(
        torch.zeros(1, num_query_token, encoder_config.hidden_size)
    )
    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    return Qformer, query_tokens

class Blip2QformerPatch(Blip2Base):
    """
    BLIP-conch first-stage model with Q-former and Conch.
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """
    def __init__(
        self,
        clip_name="conch",
        num_query_token=16,
        cross_attention_freq=2,
        embed_dim=256,
        pretrain_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        max_txt_len = 256,
    ):
        super().__init__()

        self.clip_name = clip_name

        self.vision_encoder, self.image_processor, self.img_embed_dim = self.load_vision_encoder() # LayerNorm is already done by conch

        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        self.vision_encoder = self.vision_encoder.eval()

        self.tokenizer = init_tokenizer(pretrain_name=pretrain_name) 
        self.Qformer, self.query_tokens = init_Qformer(num_query_token, self.img_embed_dim, cross_attention_freq, pretrain_name)

        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

    def load_vision_encoder(self):
        print("vision_encoder loading ...")

        clip_path = clip_path_map(self.clip_name)
        if self.clip_name=="pathclip-base":
            vision_encoder, _, image_processor = open_clip.create_model_and_transforms('ViT-B-16', pretrained=clip_path, force_quick_gelu=True)
            img_embed_dim = 512
            vision_encoder.visual.output_tokens = True
        elif self.clip_name=="conch": 
            from conch.open_clip_custom import create_model_from_pretrained
            # 448x448
            vision_encoder, image_processor = create_model_from_pretrained('conch_ViT-B-16', clip_path)
            img_embed_dim = 512
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
            img_embed_dim = 1024
        else:
            raise Exception("wrong clip")
        return vision_encoder, image_processor, img_embed_dim

    def _split_and_pad(self, image_embeds, p_num):
        """
        Split the image_embeds tensor into k lists based on p_num and pad them to the max length.
        Also generate the attention_mask.
        
        Args:
        - image_embeds (torch.Tensor): The Nx512 tensor.
        - p_num (list): List containing the number of embeddings for each segment.
        
        Returns:
        - padded_lists (list): List of k tensors, each padded to the max length.
        - attention_mask (torch.Tensor): Attention mask tensor with 1s for data positions and 0s for padding.
        """
        # Ensure p_num sums to N
        assert sum(p_num) == image_embeds.size(0), "p_num does not sum to the number of embeddings"
        
        # Split the image_embeds tensor
        start = 0
        split_lists = []
        for num in p_num:
            split_lists.append(image_embeds[start : start + num])
            start += num
        
        # Find the max length
        max_length = max(p_num)
        
        # Pad the lists and create attention masks
        padded_lists = []
        attention_masks = []
        for tensor in split_lists:
            length = tensor.size(0)
            padding = max_length - length
            padded_tensor = torch.cat([tensor, torch.zeros((padding, tensor.size(1))).to(tensor.device)], dim=0) # max_length x 512
            padded_lists.append(padded_tensor)
            
            # Create the attention mask
            attention_mask = torch.cat([torch.ones(length), torch.zeros(padding)], dim=0).to(tensor.device)
            attention_masks.append(attention_mask)
        
        # Stack the padded tensors and attention masks to create the final tensors
        padded_tensors = torch.stack(padded_lists, dim=0) # batch x max_length x 512
        attention_mask_tensor = torch.stack(attention_masks, dim=0).long() # batch x max_length
        
        return padded_tensors, attention_mask_tensor
    
    def forward(self, **kwargs):

        text = kwargs["text"]
        image = kwargs["image"]
        p_num = kwargs["patch_num"]

        with torch.inference_mode():
            if self.clip_name == 'uni':
                image_embeds = self.vision_encoder(image)
            elif self.clip_name == 'conch':
                image_embeds = self.vision_encoder.encode_image(image, normalize=False, proj_contrast=False)
            else: 
                image_embeds = self.vision_encoder.encode_image(image, normalize=False)[0] # no proj_contrast=False for clip

        image_embeds, image_atts = self._split_and_pad(image_embeds, p_num)

        image_embeds = image_embeds.to(image.device).to(torch.bfloat16) # batch x max_length x 512
        image_atts = image_atts.to(image.device).to(torch.bfloat16) # batch x max_length
        
        # image_embeds = image_embeds.unsqueeze(1) # batch,512 -> batch,1,512
        
        # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device) # batch,1

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1).to(torch.bfloat16)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats = F.normalize(self.vision_proj(query_output.last_hidden_state), dim=-1)

        text_tokens = self.tokenizer(
                                    text,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.max_txt_len,
                                    return_tensors="pt",
                                    ).to(image.device)

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )

        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

        ###============== Image-text Contrastive ===================###
        image_feats_all = concat_all_gather(
            image_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        rank = dist.get_rank()
        bs = image_embeds.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            image.device
        )
                
        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        ###============== Image-text Matching ===================###
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        image_embeds_world = all_gather_with_grad(image_embeds)
        with torch.no_grad():
            sim_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
            sim_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)            
                
            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_i2t = F.softmax(sim_i2t, dim=1)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat([text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0)  # pos, pos, neg
        text_atts_all = torch.cat([text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg], dim=0)

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(image.device)
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        image_embeds_all = torch.cat([image_embeds, image_embeds_neg, image_embeds], dim=0)  # pos, neg, pos
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(image.device)

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(image.device)
        loss_itm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss

        return BlipOutput(
            loss=loss_itc + loss_itm + loss_lm,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
        )

class Blip2QformerPathInstruct(Blip2QformerPatch):
    def __init__(self,  clip_name="conch",
                        num_query_token=16,
                        cross_attention_freq=2,
                        pretrain_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                        llm_requires_grad=False, 
                        load_in_8bit=False, 
                        load_in_4bit=False, 
                        llm_name = 'meta-llama/Meta-Llama-3-8B-Instruct', 
                        trust_remote_code=False, token=True, 
                        llm_tokenizer=None, image_token_id=None,
                        data_cache_dir = '~/.cache',
                        max_txt_len = 256):
        
        nn.Module.__init__(self)

        self.clip_name = clip_name

        self.vision_encoder, self.image_processor, self.img_embed_dim = self.load_vision_encoder() # LayerNorm is already done by conch

        # Freeze the vision_encoder parameters
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        self.vision_encoder = self.vision_encoder.eval()

        self.bert_tokenizer = init_tokenizer(pretrain_name=pretrain_name) 
        self.Qformer, self.query_tokens = init_Qformer(num_query_token, self.img_embed_dim, cross_attention_freq, pretrain_name)
        self.Qformer = self.Qformer.to(torch.bfloat16)
        self.Qformer.resize_token_embeddings(len(self.bert_tokenizer))

        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.llm_tokenizer = llm_tokenizer
        self.data_cache_dir = data_cache_dir
        self.llm = self.load_llm(load_in_8bit, load_in_4bit, llm_name, trust_remote_code, token)
        self.embedding_layer = self.llm.get_input_embeddings()

        self.resampler_layer = nn.Sequential(
                                            nn.LayerNorm(self.Qformer.config.hidden_size),
                                            nn.Linear(self.Qformer.config.hidden_size, self.llm.config.hidden_size, bias=False),
                                            nn.ReLU(),
                                            nn.Dropout(0.25),
                                            )
        
        self.config = self.llm.config
        self.image_token_id = image_token_id
        self.max_txt_len = max_txt_len

        # Control whether the LLM parameters are trainable
        for param in self.llm.parameters():
            param.requires_grad = llm_requires_grad


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
                    cache_dir = self.data_cache_dir,
                )
        llm.resize_token_embeddings(len(self.llm_tokenizer))
        return llm
        

    def get_fusion_embedding(self, input_ids, query_token):
        token_embs = self.embedding_layer(input_ids) # text token
        image_token_emb = self.embedding_layer(torch.tensor(self.image_token_id).to(query_token.device)) # special token for <Image>
        batch_image_token_emb = image_token_emb.repeat(query_token.size(0), 1, 1)
        fusion_embs =  torch.cat((batch_image_token_emb, query_token, token_embs), dim=1)
        return fusion_embs

    def pad_attention_fusion(self, new_seq_len, need_pad_seq):
        padd_len = new_seq_len - need_pad_seq.size(1)
        bz = need_pad_seq.size(0)
        generated_pad = torch.ones((bz, padd_len), dtype=need_pad_seq.dtype).to(need_pad_seq.device)
        paded_seq = torch.cat((generated_pad, need_pad_seq), dim=1)
        return paded_seq
    
    def pad_label_fusion(self, new_seq_len, labels):
        padd_len = new_seq_len - labels.size(1)
        bz = labels.size(0)
        generated_pad = torch.ones((bz, padd_len), dtype=labels.dtype).fill_(-100).to(labels.device)
        paded_seq = torch.cat((generated_pad, labels), dim=1)
        return paded_seq

    def generate(self, *args, **kwargs):
            generation_config = GenerationConfig(
                max_length=512,
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
                image = kwargs["image"].to(torch.bfloat16)
                text_input = kwargs["text_input"]
                llm_input_ids = kwargs["input_ids"].to(image.device)
                llm_input_attention_mask = kwargs["attention_mask"].to(image.device)
                p_num = kwargs["patch_num"]

                image_embeds = self.vision_encoder.encode_image(image, normalize=False, proj_contrast=False)
                image_embeds, image_atts = self._split_and_pad(image_embeds, p_num)

                image_embeds = image_embeds.to(image.device).to(torch.bfloat16) # batch x max_length x 512
                image_atts = image_atts.to(image.device) # batch x max_length

                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1).to(torch.bfloat16) # learnable query
                
                text_Qformer = self.bert_tokenizer(
                                    text_input,
                                    padding='longest',
                                    truncation=True,
                                    max_length=self.max_txt_len,
                                    return_tensors="pt",
                                ).to(image.device)
                
                query_atts = torch.ones(query_tokens.size()[:-1]).to(image.device)
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

                query_output = self.Qformer.bert(
                                    text_Qformer.input_ids,
                                    attention_mask=Qformer_atts,
                                    query_embeds=query_tokens,
                                    encoder_hidden_states=image_embeds,
                                    encoder_attention_mask=image_atts,
                                    return_dict=True,
                                )

                llm_query_input = self.resampler_layer(query_output.last_hidden_state[:,:query_tokens.size(1),:])

                fusion_embs = self.get_fusion_embedding(llm_input_ids, llm_query_input)

                fusion_attention_mask = self.pad_attention_fusion(fusion_embs.size(1), llm_input_attention_mask)
                
                res = self.llm.generate(inputs_embeds=fusion_embs, attention_mask=fusion_attention_mask, generation_config=generation_config)

            generate_list = []
            for item in res:
                generation = self.llm_tokenizer.decode(item, skip_special_tokens=True)
                generate_list.append(generation)

            return generate_list

    def forward(self, *args, **kwargs):

        text = kwargs["text_input"] # original text for inupt, i.e. question
        image = kwargs["image"]
        p_num = kwargs["patch_num"]

        llm_input_ids = kwargs["input_ids"].to(image.device) # ids for text_output
        llm_input_attention_mask = kwargs["attention_mask"].to(image.device) # attention mask for text_output
        labels = kwargs["labels"].to(image.device)  # ids for text_output

        with torch.inference_mode():
            if self.clip_name == 'uni':
                image_embeds = self.vision_encoder(image)
            elif self.clip_name == 'conch':
                image_embeds = self.vision_encoder.encode_image(image, normalize=False, proj_contrast=False)
            else: 
                image_embeds = self.vision_encoder.encode_image(image, normalize=False)[0] # no proj_contrast=False for clip

        image_embeds, image_atts = self._split_and_pad(image_embeds, p_num)

        image_embeds = image_embeds.to(image.device).to(torch.bfloat16) # batch x max_length x 512
        image_atts = image_atts.to(image.device) # batch x max_length

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1).to(torch.bfloat16) # learnable query
        
        text_Qformer = self.bert_tokenizer(
                            text,
                            padding='longest',
                            truncation=True,
                            max_length=self.max_txt_len,
                            return_tensors="pt",
                        ).to(image.device)
        
        query_atts = torch.ones(query_tokens.size()[:-1]).to(image.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

        query_output = self.Qformer.bert(
                            text_Qformer.input_ids,
                            attention_mask=Qformer_atts,
                            query_embeds=query_tokens,
                            encoder_hidden_states=image_embeds,
                            encoder_attention_mask=image_atts,
                            return_dict=True,
                        )

        llm_query_input = self.resampler_layer(query_output.last_hidden_state[:,:query_tokens.size(1),:])

        fusion_embs = self.get_fusion_embedding(llm_input_ids, llm_query_input)

        fusion_attention_mask = self.pad_attention_fusion(fusion_embs.size(1), llm_input_attention_mask)

        labels = self.pad_label_fusion(fusion_embs.size(1), labels)

        output = self.llm(inputs_embeds=fusion_embs, attention_mask=fusion_attention_mask, labels=labels)

        return output