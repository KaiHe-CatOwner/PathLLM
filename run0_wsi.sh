#!/bin/bash
#SBATCH --account=phwq4930-renal-canc
#SBATCH --qos=epsrc
#SBATCH --gpus-per-task 2
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-gpu 36
#SBATCH --nodes 1
#SBATCH --time 3-0:0:0
#SBATCH --mem 256G
#SBATCH --constraint=a100_80
#SBATCH -o ./logs/slurm_wsi_abmil_s2.%N.%j.out # STDOUT
#SBATCH -e ./logs/slurm_wsi_abmil_s2.%N.%j.err # STDERR

module purge
module load bask-apps/live
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

source /bask/projects/p/phwq4930-renal-canc/Zeyu/pyvenv/pathllm/bin/activate

export WANDB_MODE=online

accelerate launch --config_file=./accelerate_configs/deepspeed_zero2.yaml run_wsi.py --max_steps 20_000 --warmup_steps 1000\
        --gpu 2 --train_batch_size 8 --eval_batch_size 2 --max_seq_length 256 --resume_from_checkpoint False \
        --output_dir ./output/WSI_ConchLlama3.1_abmil_QA_Stage2_newtoken --agg_strategy abmil --embed_dim 512\
        --llm_name meta-llama/Meta-Llama-3.1-8B-Instruct \
        --dataset_name_list CNX-PathLLM/TCGA-WSI-CloseQA-Balanced,CNX-PathLLM/GTEx-WSI-CloseQA-Balanced,CNX-PathLLM/TCGA-WSI-OpenQA,CNX-PathLLM/GTEx-WSI-OpenQA\
        --data_cache_dir "/bask/projects/p/phwq4930-renal-canc/Zeyu/PathVLM/.cache" \
        --fea_root /bask/homes/a/asiw9691/PathVLM/WSI_Dataset/Conch \
        --n_heads 32,16,8 --llm_requires_grad True

        # --use_peft False --peft_lora_r 8  # Conch_CC GMM_PT
# --use_peft True --peft_lora_r 8 CNX-PathLLM/TCGA-WSI-Description-4onew,CNX-PathLLM/TCGA-WSI-Description-4omini,CNX-PathLLM/GTEx-WSI-Description
# accelerate launch --config_file=/raid/hpc/hekai/WorkShop/My_project/PathLLM_new/accelerate_configs/deepspeed_zero2.yaml  run.py --gpu 6 --train_batch_size 16   --eval_batch_size 16 --max_seq_length 256

# accelerate launch --config_file=/raid/hpc/hekai/WorkShop/My_project/PathLLM_new/accelerate_configs/deepspeed_zero3.yaml  run.py