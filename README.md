# PathLLM# PathLLM

## Longnet ##

(1) --vision_adaptor False --hierachical_adaptor True
(2) --vision_adaptor False --hierachical_adaptor False
(3) --vision_adaptor True --hierachical_adaptor True

```

--vision_adaptor False (vision-query-question interaction)

--vision_adaptor True (vision-query interaction)

--hierachical_adaptor False (same adaptor for all level)

--hierachical_adaptor True (different adaptors for different levels)

```

### Train Step 1 ###
```
accelerate launch --config_file=./accelerate_configs/deepspeed_zero2.yaml run_wsi.py --max_steps 20_000 --warmup_steps 1000\
        --gpu 2 --train_batch_size 4 --eval_batch_size 2 --max_seq_length 512 \
        --agg_strategy longnet --embed_dim 512 --vision_adaptor False --hierachical_token True --hierachical_adaptor True\
        --n_heads 32,16,8 --llm_requires_grad True --resume_from_checkpoint False \
        --llm_name meta-llama/Meta-Llama-3.1-8B-Instruct \
        --dataset_name_list CNX-PathLLM/TCGA-WSI-Description-4onew,CNX-PathLLM/TCGA-WSI-Description-4omini,CNX-PathLLM/GTEx-WSI-Description \
        --data_cache_dir "/bask/projects/p/phwq4930-renal-canc/Zeyu/PathVLM/.cache" \
        --fea_root /bask/homes/a/asiw9691/PathVLM/WSI_Dataset/Conch \
        --output_dir ./output/WSI_ConchLlama3.1_longnet_QA_Stage1_hmltoken 
```

### Train Step 2 ###
```
accelerate launch --config_file=./accelerate_configs/deepspeed_zero2.yaml run_wsi.py --max_steps 20_000 --warmup_steps 1000\
        --gpu 2 --train_batch_size 4 --eval_batch_size 2 --max_seq_length 256 \ 
        --agg_strategy longnet --embed_dim 512 --vision_adaptor False --hierachical_token True --hierachical_adaptor True\
        --n_heads 32,16,8 --llm_requires_grad True --resume_from_checkpoint False\
        --llm_name meta-llama/Meta-Llama-3.1-8B-Instruct \
        --dataset_name_list CNX-PathLLM/TCGA-WSI-CloseQA-Balanced,CNX-PathLLM/GTEx-WSI-CloseQA-Balanced,CNX-PathLLM/TCGA-WSI-OpenQA,CNX-PathLLM/GTEx-WSI-OpenQA\
        --data_cache_dir "/bask/projects/p/phwq4930-renal-canc/Zeyu/PathVLM/.cache" \
        --fea_root /bask/homes/a/asiw9691/PathVLM/WSI_Dataset/Conch \
        --output_dir ./output/WSI_ConchLlama3.1_longnet_QA_Stage2_hmltoken \
        --ckpt_path path/to/ckpt.bin/of/step1
```

### Train Step 3 ###
To continue training with the specific detailed BRCA dataset!

### Test ###
#### description ####
```
python test_wsi.py --max_seq_length 128 --batch_size 2 --select_data_num -1 --eval_sample_size -1 --n_heads 32,16,8 --agg_strategy abmil \
                    --embed_dim 512 --vision_adaptor False --hierachical_token True --hierachical_adaptor True\
                    --llm_name meta-llama/Meta-Llama-3.1-8B-Instruct \
                    --shuffle False --data_cache_dir /bask/projects/p/phwq4930-renal-canc/Zeyu/PathVLM/.cache \
                    --dataset_name_list CNX-PathLLM/TCGA-WSI-Description-4onew,CNX-PathLLM/TCGA-WSI-Description-4omini,CNX-PathLLM/GTEx-WSI-Description \
                    --fea_root /bask/homes/a/asiw9691/PathVLM/WSI_Dataset/Conch \
                    --ckpt_path /bask/homes/a/asiw9691/PathVLM/source/PathLLM/output/WSI_ConchLlama3.1_abmil_QA_Stage1_hmltoken/ckpt20000.bin\
                    --results_save_path /bask/homes/a/asiw9691/PathVLM/source/PathLLM/output/WSI_ConchLlama3.1_abmil_QA_Stage1_hmltoken/ckpt20000.csv
```
#### Q&A ####
```
python test_wsi.py --max_seq_length 128 --batch_size 2 --select_data_num -1 --eval_sample_size -1 --n_heads 32,16,8 --agg_strategy abmil \
                    --embed_dim 512 --vision_adaptor False --hierachical_token True --hierachical_adaptor True\
                    --llm_name meta-llama/Meta-Llama-3.1-8B-Instruct \
                    --shuffle False --data_cache_dir /bask/projects/p/phwq4930-renal-canc/Zeyu/PathVLM/.cache \
                    --dataset_name_list CNX-PathLLM/TCGA-WSI-CloseQA-Balanced,CNX-PathLLM/GTEx-WSI-CloseQA-Balanced,CNX-PathLLM/TCGA-WSI-OpenQA,CNX-PathLLM/GTEx-WSI-OpenQA \
                    --fea_root /bask/homes/a/asiw9691/PathVLM/WSI_Dataset/Conch \
                    --ckpt_path /bask/homes/a/asiw9691/PathVLM/source/PathLLM/output/WSI_ConchLlama3.1_abmil_QA_Stage2_hmltoken/ckpt20000.bin\
                    --results_save_path /bask/homes/a/asiw9691/PathVLM/source/PathLLM/output/WSI_ConchLlama3.1_abmil_QA_Stage2_hmltoken/ckpt20000.csv
```

### GPT_EVAL ###

```
python GPT_evaluation.py --type open --filename output/WSI_ConchLlama3.1_abmil_QA_Stage12_newtoken/ckpt9500_open.csv
```
