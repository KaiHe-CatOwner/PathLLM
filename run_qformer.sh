#!/bin/bash
#SBATCH --account=phwq4930-gbm
#SBATCH --qos=epsrc
#SBATCH --gpus-per-task 2
#SBATCH --tasks-per-node 1
#SBATCH --nodes 1
#SBATCH --cpus-per-gpu 36
#SBATCH --time 12:0:0
#SBATCH --mem 128G
#SBATCH --constraint=a100_40
#SBATCH -o ./logs/slurm.%N.%j.out # STDOUT
#SBATCH -e ./logs/slurm.%N.%j.err # STDERR

module purge
module load bask-apps/live
module load GCCcore/11.3.0 Python/3.10.4
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0


# source /bask/projects/p/phwq4930-gbm/Zeyu/pyvenv/medfla2/bin/activate
source /bask/projects/p/phwq4930-gbm/Zeyu/pyvenv/pathllmGZY/bin/activate
export WANDB_MODE=online

accelerate launch --config_file=./accelerate_configs/deepspeed_zero2.yaml run_qformer.py --learning_rate 1e-4 --max_steps 200_000 --warmup_ratio 0.01\
        --gpu 2 --train_batch_size 64 --eval_batch_size 2 --max_seq_length 256 --resume_from_checkpoint False --output_dir ./output/Conch_Bert_PatchPre\
        --llm_name meta-llama/Meta-Llama-3-8B-Instruct --clip_name conch --bert_name microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext\
        --dataset_name_list CNX-PathLLM/Pathcap,CNX-PathLLM/PubMedPath,CNX-PathLLM/TwitterPath,CNX-PathLLM/CleanedTextData\
        --data_cache_dir /bask/projects/p/phwq4930-gbm/Zeyu/PathVLM/.cache

# accelerate launch --config_file=/raid/hpc/hekai/WorkShop/My_project/PathLLM_new/accelerate_configs/deepspeed_zero2.yaml  run.py --gpu 6 --train_batch_size 16   --eval_batch_size 16 --max_seq_length 256

# accelerate launch --config_file=/raid/hpc/hekai/WorkShop/My_project/PathLLM_new/accelerate_configs/deepspeed_zero3.yaml  run.py
