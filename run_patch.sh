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


# source /home/z/zeyugao/pyvenv/pathllmGZY/bin/activate
source /bask/projects/p/phwq4930-renal-canc/Zeyu/pyvenv/pathllmGZY/bin/activate
export WANDB_MODE=online
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=600

accelerate launch --config_file=./accelerate_configs/deepspeed_zero2.yaml run_patch.py --max_steps 20_000\
        --gpu 2 --train_batch_size 24 --eval_batch_size 2 --max_seq_length 256 --resume_from_checkpoint True \
        --llm_requires_grad False \
        --data_cache_dir /bask/projects/p/phwq4930-renal-canc/Zeyu/PathVLM/.cache\
        --output_dir /bask/homes/a/asiw9691/PathVLM/source/PathLLM/output/Conch_Llama3.1_Stage1\
        --llm_name meta-llama/Meta-Llama-3.1-8B-Instruct --clip_name conch\
        --dataset_local_paths /bask/homes/a/asiw9691/PathVLM/VLM_dataset/pretrain_data_all\
        --dataset_name_list CNX-PathLLM/Pathcap,CNX-PathLLM/PubMedPath,CNX-PathLLM/TwitterPath,CNX-PathLLM/CleanedTextData
        # --ckpt_path /bask/homes/a/asiw9691/PathVLM/source/PathLLM/output/Conch_Llama3.1_Stage1/checkpoint-15000/ckpt15000.bin

# accelerate launch --config_file=/raid/hpc/hekai/WorkShop/My_project/PathLLM_new/accelerate_configs/deepspeed_zero2.yaml  run.py --gpu 6 --train_batch_size 16   --eval_batch_size 16 --max_seq_length 256

# accelerate launch --config_file=/raid/hpc/hekai/WorkShop/My_project/PathLLM_new/accelerate_configs/deepspeed_zero3.yaml  run.py 

#123 --account=phwq4930-gbm
#123 --qos=epsrc
#123 --gpus-per-task 2
#123 --tasks-per-node 1
#123 --nodes 1
#123 --time 24:0:0
#123 --mem 256G
#123 --constraint=a100_80
#123 -o ./logs/slurm.%N.%j.out # STDOUT
#123 -e ./logs/slurm.%N.%j.err # STDERR

