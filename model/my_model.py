# -*- coding:utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
# import gigapath.slide_encoder_old as slide_encoder
import gigapath.slide_encoder as slide_encoder
# from gigapath.pipeline import run_inference_with_slide_encoder
from torchvision import transforms
import timm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from transformers import BertModel, BertConfig
from accelerate import Accelerator
from utils.utils import clip_path_map

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.proj1 = nn.Linear(in_features, rank, bias=False)  # 从 in_features 降到 rank
        self.proj2 = nn.Linear(rank, out_features, bias=True)  # 从 rank 映射到 out_features

    def forward(self, x):
        return self.proj2(self.proj1(x))

class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, heads = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, heads)

    def forward(self, x, mask):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x heads
        A[mask==0] = 1e-9
        return A
    
class XAttentionModel(nn.Module):
    def __init__(self, llm_dim, embed_dim, num_layers=2, num_heads=16):
        super(AttentionLayer, self).__init__()

        print("#####Vision-Text Interation Qformer######")

        self.num_layers = num_layers
        
        # 使用 BertModel 创建 self-attention 层
        config = BertConfig(
            hidden_size=embed_dim,
            num_attention_heads=num_heads,
            num_hidden_layers=1  # 只使用一层 Bert
        )
        # text embeds dimensional reduce
        self.reduce = nn.Sequential(
            nn.Linear(llm_dim, embed_dim),  # reduction to embed_dim
            nn.ReLU()                   # Activation function
        )

        self.self_attention = nn.ModuleList([BertModel(config) for _ in range(num_layers)])
        self.cross_attention = nn.ModuleList([nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True) for _ in range(num_layers)])
    
    def forward(self, input_tensor, context_tensor, instruct_tensor, self_attention_mask=None, key_padding_mask=None):
        """
        input_tensor: Input feature tensor for self-attention processing
        context_tensor: Context feature tensor for cross-attention processing
        instruct_tensor: Instruction tensor to be concatenated with input_tensor for self-attention
        query_length: Length of the query to retain for cross-attention
        self_attention_mask: Self-attention mask (1 indicates valid position, 0 indicates padding)
        key_padding_mask: Cross-attention key padding mask (True -> no attention)
        """
        query_length = input_tensor.shape[1]
        instruct_tensor = self.reduce(instruct_tensor)

        if self_attention_mask is not None:
            # instruct_tensor requires a mask of the same size filled with ones
            input_mask_extension = torch.ones((input_tensor.shape[0], input_tensor.shape[1]), dtype=self_attention_mask.dtype, device=self_attention_mask.device)
            self_attention_mask = torch.cat((input_mask_extension, self_attention_mask), dim=-1)

        for i in range(self.num_layers):
            combined_tensor = torch.cat((input_tensor, instruct_tensor), dim=1)
            self_attn_output = self.self_attention[i](
                inputs_embeds=combined_tensor,
                attention_mask=self_attention_mask
            ).last_hidden_state
            query = self_attn_output[:, :query_length, :] # keep query vector only
            # Cross-Attention
            input_tensor, _ = self.cross_attention[i](query=query, key=context_tensor, value=context_tensor, key_padding_mask=key_padding_mask)

        # combined_tensor = torch.cat((cross_attn_output, instruct_tensor), dim=1)
        # self_attn_output = self.self_attention1(
        #     inputs_embeds=combined_tensor,
        #     attention_mask=self_attention_mask
        # ).last_hidden_state
        # query = self_attn_output[:, :query_length, :] # keep query vector only
        # cross_attn_output, _ = self.cross_attention1(query=query, key=context_tensor, value=context_tensor, key_padding_mask=key_padding_mask)
        
        return input_tensor

class PPathVLM(nn.Module):
    def __init__(self, llm_requires_grad, clip_name, load_in_8bit, load_in_4bit, llm_name, 
                 trust_remote_code, token, tokenizer, image_token_id, data_cache_dir='~/.cache'):
        nn.Module.__init__(self)

        self.clip_name = clip_name
        self.data_cache_dir = data_cache_dir

        self.vision_encoder, self.image_processor, self.embed_dim = self.load_vision_encoder()

        self.llm_tokenizer = tokenizer
        self.llm = self.load_llm(load_in_8bit, load_in_4bit, llm_name, trust_remote_code, token)
        self.embedding_layer = self.llm.get_input_embeddings()

        # self.fusion_layer_S = nn.Linear(self.embed_dim, self.llm.config.hidden_size)
        self.resampler_layer = nn.Sequential(
                                            nn.LayerNorm(self.embed_dim),
                                            nn.Linear(self.embed_dim, self.llm.config.hidden_size, bias=False),
                                            nn.ReLU(),
                                            nn.Dropout(0.25),
                                            )
        # self.fusion_layer_E = torch.nn.TransformerEncoderLayer(self.llm.config.hidden_size, 8,  batch_first=True)
        
        self.config = self.llm.config
        self.image_token_id = image_token_id

        # Freeze the vision_encoder parameters
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # Control whether the LLM parameters are trainable
        for param in self.llm.parameters():
            param.requires_grad = llm_requires_grad

        # self.vision_encoder.requires_grad = False
        # self.llm.requires_grad = llm_requires_grad

    def print_parameter_counts(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Total number of parameters: {total_params}")
        print(f"Number of trainable parameters: {trainable_params}")

    def print_llm_parameters(self, num_params=5):
            """Print a few parameters of the LLM to check requires_grad status."""
            count = 0
            for name, param in self.llm.named_parameters():
                if count >= num_params:
                    break
                print(f"Parameter name: {name}")
                print(f"Parameter requires_grad: {param.requires_grad}")
                print(f"Parameter value: {param.data.flatten()[:5]}")  # Print first 5 values
                print("-" * 50)
                count += 1

    def print_vision_parameters(self, num_params=5):
            """Print a few parameters of the Vision Encoder to check requires_grad status."""
            count = 0
            for name, param in self.vision_encoder.visual.trunk.blocks.named_parameters():
                if count >= num_params:
                    break
                print(f"Parameter name: {name}")
                print(f"Parameter requires_grad: {param.requires_grad}")
                print(f"Parameter value: {param.data.flatten()[:5]}")  # Print first 5 values
                print("-" * 50)
                count += 1

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
            # torch_dtype = torch.bfloat16


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
    
    def generate(self, *args, **kwargs):
        generation_config = GenerationConfig(
                max_length=100,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
                num_return_sequences=1,
                repetition_penalty=1.0,
                do_sample=True,
                pad_token_id=self.llm_tokenizer.eos_token_id,
                bos_token_id=self.llm_tokenizer.bos_token_id,
            )
        
        with torch.no_grad():
            image = kwargs["image"]
            p_num = kwargs["patch_num"]
            input_ids = kwargs["input_ids"].to(image.device) # ids for text_output
            attention_mask = kwargs["attention_mask"].to(image.device) # attention mask for text_output

            with torch.inference_mode():
                if self.clip_name == 'uni':
                    image_embeds = self.vision_encoder(image)
                elif self.clip_name == 'conch':
                    image_embeds = self.vision_encoder.encode_image(image, normalize=False, proj_contrast=False)
                else: 
                    image_embeds = self.vision_encoder.encode_image(image, normalize=False)[0] # no proj_contrast=False for clip

            image_embeds, image_atts = self._split_and_pad(image_embeds, p_num)

            image_embeds = image_embeds.to(image.device)# .to(torch.bfloat16)# batch x max_length x 512
            image_atts = image_atts.to(image.device) # batch x max_length
            attention_mask = torch.cat([image_atts, attention_mask], dim=1)
        
            fusion_embs = self.get_fusion_embedding(input_ids, image_embeds)
            attention_mask = self.pad_attention_fusion(fusion_embs.size(1), attention_mask)
            res = self.llm.generate(inputs_embeds=fusion_embs, attention_mask=attention_mask, generation_config=generation_config)

        generate_list = []
        for item in res:
            generation = self.llm_tokenizer.decode(item, skip_special_tokens=True)
            generate_list.append(generation)
        return generate_list
    
    def pad_attention_fusion(self, new_seq_len, need_pad_seq):
        padd_len = new_seq_len - need_pad_seq.size(1)
        bz = need_pad_seq.size(0)

        generated_pad = torch.ones((bz, padd_len), dtype=need_pad_seq.dtype).to(need_pad_seq.device)

        # if self.llm_tokenizer.padding_side == "right":
        #     paded_seq = torch.cat((need_pad_seq, generated_pad), dim=1)
        # else:
        paded_seq = torch.cat((generated_pad, need_pad_seq), dim=1)

        return paded_seq
    
    def pad_label_fusion(self, new_seq_len, labels):
        padd_len = new_seq_len - labels.size(1)
        bz = labels.size(0)

        generated_pad = torch.ones((bz, padd_len), dtype=labels.dtype).fill_(-100).to(labels.device)
        paded_seq = torch.cat((generated_pad, labels), dim=1)

        return paded_seq

    def get_fusion_embedding(self, input_ids, image_embs):
        token_embs = self.embedding_layer(input_ids)

        mapped_image_embs = self.resampler_layer(image_embs) # shape (bz, 512) - (bz, 4096)
        # mapped_image_embs = self.fusion_layer_E(mapped_image_embs) # shape (bz, 4096)
        # mapped_image_embs = mapped_image_embs.unsqueeze(1) # shape (bz, 1, llm_hidden_state)
        
        image_token_emb = self.embedding_layer(torch.tensor(self.image_token_id).to(mapped_image_embs.device))
        batch_image_token_emb = image_token_emb.repeat(mapped_image_embs.size(0), 1, 1)
        fusion_embs =  torch.cat((batch_image_token_emb, mapped_image_embs, token_embs), dim=1)
        return fusion_embs
    
    def forward(self, *args, **kwargs):
        image = kwargs["image"]
        p_num = kwargs["patch_num"]
        input_ids = kwargs["input_ids"].to(image.device) # ids for text_output
        attention_mask = kwargs["attention_mask"].to(image.device) # attention mask for text_output
        labels = kwargs["labels"].to(image.device)  # ids for text_output

        with torch.inference_mode():
            if self.clip_name == 'uni':
                image_embeds = self.vision_encoder(image)
            elif self.clip_name == 'conch':
                image_embeds = self.vision_encoder.encode_image(image, normalize=False, proj_contrast=False)
            else: 
                image_embeds = self.vision_encoder.encode_image(image, normalize=False)[0] # no proj_contrast=False for clip

        image_embeds, image_atts = self._split_and_pad(image_embeds, p_num)

        image_embeds = image_embeds.to(image.device).to(torch.bfloat16)# batch x max_length x 512
        image_atts = image_atts.to(image.device) # batch x max_length
        attention_mask = torch.cat([image_atts, attention_mask], dim=1)
    
        fusion_embs = self.get_fusion_embedding(input_ids, image_embeds)
        attention_mask = self.pad_attention_fusion(fusion_embs.size(1), attention_mask)
        labels = self.pad_label_fusion(fusion_embs.size(1), labels)

        output = self.llm(inputs_embeds=fusion_embs, attention_mask=attention_mask, labels=labels)
        return output
    
class WPathVLM(PPathVLM):
    def __init__(self, llm_requires_grad, load_in_8bit, load_in_4bit, llm_name, 
                 trust_remote_code, token, tokenizer, image_token_id,
                 n_heads='32,16,8', n_level=3, embed_dim=512, 
                 agg_strategy='abmil', hierachical_token=True, hierachical_adaptor=True,
                 data_cache_dir = '~/.cache'):
        
        nn.Module.__init__(self)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.llm_tokenizer = tokenizer
        self.data_cache_dir = data_cache_dir
        self.llm = self.load_llm(load_in_8bit, load_in_4bit, llm_name, trust_remote_code, token)
        self.embedding_layer = self.llm.get_input_embeddings()
        self.n_heads = [int(n_head) for n_head in n_heads.split(',')]
        self.n_level = n_level
        self.agg_strategy = agg_strategy
        self.embed_dim = embed_dim
        self.config = self.llm.config
        self.image_token_id = image_token_id # ['<|Image|>', '<|High|>', '<|`Mid`|>', '<|Low|>']
        self.hierachical_token = hierachical_token
        self.hierachical_adaptor = hierachical_adaptor
        self.patch_size_dict = {0:1024.0, 1:2048.0, 2:4096.0}

        if self.hierachical_adaptor:
            self.adaptor_level = n_level
        else:
            self.adaptor_level = 1

        if self.agg_strategy == 'abmil':
            size = [embed_dim, int(embed_dim/2)]
        # attention could be splitted as level based
            self.att_net = nn.ModuleList([
                Attn_Net_Gated(L=size[0], D=size[1], dropout=True, heads=self.n_heads[i]) 
                for i in range(self.n_level)
            ])

        # LongNet + CrossAttention
        if self.agg_strategy == "longnet":
            self.reduce = nn.Sequential(
                LowRankLinear(self.llm.config.hidden_size, self.embed_dim, rank=128),  # rank=128
                nn.ReLU(),
                )
            self.query = [nn.Parameter(torch.zeros(1, self.n_heads[i], embed_dim)) for i in range(self.n_level)]
            self.longnet_encoder_list = nn.ModuleList([
                slide_encoder.create_model(pretrained="", model_arch="gigapath_slide_enc2l512d", tile_size=self.patch_size_dict[level])
                for level in range(self.adaptor_level)]).to(torch.bfloat16)

        # CrossAttention
        if self.agg_strategy == "qformer":
            self.query = [nn.Parameter(torch.zeros(1, self.n_heads[i], embed_dim)) for i in range(self.n_level)]
            self.qformer_encoder_list = nn.ModuleList([XAttentionModel(self.llm.config.hidden_size, embed_dim) for _ in range(self.adaptor_level)]).to(torch.bfloat16)

        if self.agg_strategy in ["qformer", "longnet"]:
            for tensor in self.query:
                nn.init.xavier_uniform_(tensor)

        self.resampler_layer = nn.Sequential(
                                            nn.LayerNorm(self.embed_dim),
                                            nn.Linear(self.embed_dim, self.llm.config.hidden_size, bias=False),
                                            nn.ReLU(),
                                            nn.Dropout(0.25),
                                        )

        # Control whether the LLM parameters are trainable
        for param in self.llm.parameters():
            param.requires_grad = llm_requires_grad

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def get_wsi_embedding(self, patch_embs, patch_masks, coords, level, input_ids_instruct=None, attention_mask_instruct=None):

        if self.hierachical_adaptor:
            adaptor_level = level
        else:
            adaptor_level = 0

        # input_ids_instruct, attention_mask_instruct for text interaction

        if self.agg_strategy == 'abmil':

            batch_size, num_patches, embedding_size = patch_embs.shape # (bz, np(cc), 512)
            # patch embedding attention part
            patch_embs_flattened = patch_embs.view(batch_size * num_patches, embedding_size) # shape (bz x np, 512)
            patch_masks_flattened = patch_masks.view(batch_size * num_patches)

            patch_attention_matrices = self.att_net[level](patch_embs_flattened, patch_masks_flattened) # shape (bz x np, n_heads)
            patch_attention_matrices = patch_attention_matrices.view(batch_size, num_patches, self.n_heads[level]) # shape (bz, np, n_heads)
            patch_attention_matrices = F.softmax(patch_attention_matrices, dim=1) # shape (bz, np, n_heads)
            # aggregated to WSI embedding
            mapped_patch_embs = patch_embs_flattened.view(batch_size, num_patches, embedding_size) # shape (bz, np, 512)
            agged_WSI_embs = patch_attention_matrices.unsqueeze(-1) * mapped_patch_embs.unsqueeze(-2)  # shape (bz, np, n_heads, 512)
            agged_WSI_embs = torch.sum(agged_WSI_embs, dim=1) # (bz, n_heads, 512)

        elif self.agg_strategy == "longnet":
            patch_size_dict = {key: torch.tensor([value]) for key, value in self.patch_size_dict.items()}
            query = self.query[level].repeat(patch_embs.shape[0], 1, 1)
            key_padding_mask = patch_masks.bool()
            instruct_embs = self.embedding_layer(input_ids_instruct)
            instruct_embs = self.reduce(instruct_embs)
            agged_WSI_embs = self.run_inference_with_slide_encoder(query.cuda().to(torch.bfloat16), patch_embs.to(torch.bfloat16), 
                                                                instruct_embs, coords.to(torch.bfloat16),
                                                                self_attention_mask=attention_mask_instruct,
                                                                key_padding_mask=~key_padding_mask,
                                                                slide_encoder_model=self.longnet_encoder_list[adaptor_level],
                                                                patch_size=patch_size_dict[level].cuda().to(torch.bfloat16)) # .to(torch.bfloat16)

        elif self.agg_strategy == "qformer":
            query = self.query[level].repeat(patch_embs.shape[0], 1, 1)
            key_padding_mask = patch_masks.bool()
            instruct_embs = self.embedding_layer(input_ids_instruct)
            agged_WSI_embs = self.qformer_encoder_list[adaptor_level](query.cuda().to(torch.bfloat16), patch_embs.to(torch.bfloat16), 
                                                                      instruct_embs.to(torch.bfloat16),
                                                                      self_attention_mask=attention_mask_instruct, 
                                                                      key_padding_mask=~key_padding_mask)
        
        else: # "sample, kmeans, gmm"
            agged_WSI_embs = patch_embs

        return agged_WSI_embs
     
    # used in LongNet+Crossattention
    def run_inference_with_slide_encoder(self, query, tile_embeds, instruct_embs, coords, slide_encoder_model, patch_size=256,
                                         self_attention_mask=None, key_padding_mask=None):
        """         
        Run inference with the slide encoder
                
        Arguments:  
        ----------  
        tile_embeds : torch.Tensor
            Tile embeddings
        coords : torch.Tensor
            Coordinates of the tiles
        slide_encoder_model : torch.nn.Module
            Slide encoder model
        """     
        if len(tile_embeds.shape) == 2:
            tile_embeds = tile_embeds.unsqueeze(0)
            coords = coords.unsqueeze(0) 
                    
        # slide_encoder_model = slide_encoder_model.cuda()
        # print("1:{}".format(torch.cuda.memory_allocated(0)))
        # slide_encoder_model = slide_encoder_model.cuda()
                      
        # print("2:{}".format(torch.cuda.memory_allocated(0)))
        # slide_encoder_model.eval()
         # run inference
        with torch.cuda.amp.autocast(dtype=torch.float16):
            # slide_embeds = slide_encoder_model(tile_embeds.cuda(), coords.cuda(), all_layer_embed=True)
            slide_embeds = slide_encoder_model(query, tile_embeds, instruct_embs, coords, patch_size,
                                               self_attention_mask=self_attention_mask, key_padding_mask=key_padding_mask)
        # outputs = {"layer_{}_embed".format(i): slide_embeds[i].cuda() for i in range(len(slide_embeds))}
        # outputs["last_layer_embed"] = slide_embeds[-1].cuda()
        # print("3:{}".format(torch.cuda.memory_allocated(0)))
        return slide_embeds

    def get_fusion_embedding(self, input_ids, agged_WSI_embs):
        
        token_embs = self.embedding_layer(input_ids)
        image_token_emb = self.embedding_layer(torch.tensor(self.image_token_id).to(agged_WSI_embs[0].device))
        batch_image_token_emb = image_token_emb.repeat(agged_WSI_embs[0].size(0), 1, 1)

        agged_WSI_embs = torch.cat(agged_WSI_embs, dim=1)
        agged_WSI_embs = self.resampler_layer(agged_WSI_embs) # (bz, n_heads, 512) -> # (bz, n_heads, 4096)
        # print("4:{}".format(torch.cuda.memory_allocated(0)))
        fusion_embs =  torch.cat((batch_image_token_emb, agged_WSI_embs, token_embs), dim=1)
        return fusion_embs

    def get_fusion_embedding_hierarchy(self, input_ids, agged_WSI_embs):
        
        text_token_embs = self.embedding_layer(input_ids)
        image_token_emb = self.embedding_layer(torch.tensor(self.image_token_id[0]).to(agged_WSI_embs[0].device))
        batch_image_token_emb = image_token_emb.repeat(agged_WSI_embs[0].size(0), 1, 1)

        mag_WSI_embs = []

        for i in range(self.n_level):
            mag_token_emb = self.embedding_layer(torch.tensor(self.image_token_id[i+1]).to(agged_WSI_embs[0].device))
            batch_mag_token_emb = mag_token_emb.repeat(agged_WSI_embs[0].size(0), 1, 1)
            agged_WSI_emb = self.resampler_layer(agged_WSI_embs[i])
            mag_WSI_emb = torch.cat((batch_mag_token_emb, agged_WSI_emb), dim=1)
            mag_WSI_embs.append(mag_WSI_emb)

        # agged_WSI_embs = torch.cat(agged_WSI_embs, dim=1)
        # agged_WSI_embs = self.resampler_layer(agged_WSI_embs) # (bz, n_heads, 512) -> # (bz, n_heads, 4096)

        mag_WSI_embs = torch.cat(mag_WSI_embs, dim=1)
        # print("4:{}".format(torch.cuda.memory_allocated(0)))
        fusion_embs =  torch.cat((batch_image_token_emb, mag_WSI_embs, text_token_embs), dim=1)
        return fusion_embs

    def generate(self, *args, **kwargs):
        generation_config = GenerationConfig(
                max_length=200,
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
            text_attention_mask = kwargs["attention_mask"]
            input_ids_instruct = kwargs["input_ids_instruct"]
            attention_mask_instruct = kwargs["attention_mask_instruct"]
            # labels = kwargs["labels"]
            agged_WSI_embs = []
            
            for level in range(self.n_level):
                patch_embs = kwargs["fea{}".format(level)].float() # embeddings for patches, fea1 (40x), fea2 (20x), fea3(10x)
                patch_attention_mask = kwargs["mask{}".format(level)] # attention masks for patches, mask1 (40x), mask2 (20x), mask3 (10x) [1, 0]->[value, empty]
                coords = kwargs["cor{}".format(level)]
                agged_WSI_embs_level = self.get_wsi_embedding(patch_embs, patch_attention_mask, coords, level,
                                                            input_ids_instruct = input_ids_instruct,
                                                            attention_mask_instruct = attention_mask_instruct)
                #agged_WSI_embs_level = self.resampler_layer(agged_WSI_embs_level) # (bz, n_heads, 512) -> # (bz, n_heads, 4096)
                agged_WSI_embs.append(agged_WSI_embs_level.float())

            if self.hierachical_token:
                fusion_embs = self.get_fusion_embedding_hierarchy(input_ids, agged_WSI_embs)
            else:
                fusion_embs = self.get_fusion_embedding(input_ids, agged_WSI_embs)
            text_attention_mask = self.pad_attention_fusion(fusion_embs.size(1), text_attention_mask)

            res = self.llm.generate(inputs_embeds=fusion_embs, attention_mask=text_attention_mask, generation_config=generation_config)

        generate_list = []
        for item in res:
            generation = self.llm_tokenizer.decode(item, skip_special_tokens=True)
            generate_list.append(generation)
        return generate_list
    
    def forward(self, *args, **kwargs):
        input_ids = kwargs["input_ids"] # ids for text
        text_attention_mask = kwargs["attention_mask"]
        input_ids_instruct = kwargs["input_ids_instruct"]
        attention_mask_instruct = kwargs["attention_mask_instruct"]
        labels = kwargs["labels"]
        agged_WSI_embs = []
        
        for level in range(self.n_level):
            patch_embs = kwargs["fea{}".format(level)] # embeddings for patches, fea1 (40x), fea2 (20x), fea3(10x)
            patch_attention_mask = kwargs["mask{}".format(level)] # attention masks for patches, mask1 (40x), mask2 (20x), mask3 (10x) [1, 0]->[value, empty]
            coords = kwargs["cor{}".format(level)]
            agged_WSI_embs_level = self.get_wsi_embedding(patch_embs, patch_attention_mask, coords, level,
                                                          input_ids_instruct = input_ids_instruct,
                                                          attention_mask_instruct = attention_mask_instruct)
            # agged_WSI_embs_level = self.resampler_layer(agged_WSI_embs_level) # (bz, n_heads, 512) -> # (bz, n_heads, 4096)
            agged_WSI_embs.append(agged_WSI_embs_level)
        
        if self.hierachical_token:
            fusion_embs = self.get_fusion_embedding_hierarchy(input_ids, agged_WSI_embs)
        else:
            fusion_embs = self.get_fusion_embedding(input_ids, agged_WSI_embs)

        text_attention_mask = self.pad_attention_fusion(fusion_embs.size(1), text_attention_mask)
        labels = self.pad_label_fusion(fusion_embs.size(1), labels)

        output = self.llm(inputs_embeds=fusion_embs, attention_mask=text_attention_mask, labels=labels)
        return output