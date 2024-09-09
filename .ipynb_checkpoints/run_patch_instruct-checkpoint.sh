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
#SBATCH -o ./logs/slurm.%N.%j.out # STDOUT
#SBATCH -e ./logs/slurm.%N.%j.err # STDERR

module purge
module load bask-apps/live
module load GCCcore/11.3.0 Python/3.10.4
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

source /bask/projects/p/phwq4930-renal-canc/Zeyu/pyvenv/pathllmGZY/bin/activate
export WANDB_MODE=online

accelerate launch --config_file=./accelerate_configs/deepspeed_zero2.yaml run_patch_instruct.py --max_steps 20_000\
        --gpu 2 --train_batch_size 18 --eval_batch_size 2 --max_seq_length 256 --resume_from_checkpoint False\
        --output_dir /bask/homes/a/asiw9691/PathVLM/source/PathLLM/output/Conch_Llama3.1_Stage2\
        --llm_name meta-llama/Meta-Llama-3.1-8B-Instruct  --clip_name conch\
        --dataset_name_list CNX-PathLLM/Pathinstruct,CNX-PathLLM/TextbookQAPair,CNX-PathLLM/YoutubeInstruct\ # CNX-PathLLM/MultiConversation
        --data_cache_dir /bask/projects/p/phwq4930-renal-canc/Zeyu/PathVLM/.cache\
        --llm_requires_grad True --use_peft True --peft_lora_r 8 \
        --ckpt_path /bask/homes/a/asiw9691/PathVLM/source/PathLLM/output/Conch_Llama3.1_Stage2_lora8/checkpoint-12500/ckpt12500.bin

# accelerate launch --config_file=/raid/hpc/hekai/WorkShop/My_project/PathLLM_new/accelerate_configs/deepspeed_zero2.yaml  run.py --gpu 6 --train_batch_size 16   --eval_batch_size 16 --max_seq_length 256

# accelerate launch --config_file=/raid/hpc/hekai/WorkShop/My_project/PathLLM_new/accelerate_configs/deepspeed_zero3.yaml  run.py 

