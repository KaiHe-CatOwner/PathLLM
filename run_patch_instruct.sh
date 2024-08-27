#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --tasks-per-node 1
#SBATCH --gres=gpu:ampere_a100:3
#SBATCH --partition=gpu
#SBATCH --time 48:0:0
#SBATCH --mem-per-cpu=3850
#SBATCH --account=su123
#SBATCH -o ./logs/slurm.%N.%j.out # STDOUT
#SBATCH -e ./logs/slurm.%N.%j.err # STDERR

module purge
# module load bask-apps/live
module load GCCcore/11.3.0 Python/3.10.4
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0


source /home/z/zeyugao/pyvenv/pathllmGZY/bin/activate
# source /bask/projects/p/phwq4930-gbm/Zeyu/pyvenv/pathllmGZY/bin/activate
export WANDB_MODE=online
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=600

accelerate launch --config_file=./accelerate_configs/deepspeed_zero2.yaml run_patch.py --max_steps 20_000\
        --gpu 3 --train_batch_size 6 --eval_batch_size 2 --max_seq_length 256 --resume_from_checkpoint True\
        --output_dir /home/shared/su123/LLM_ckpt/Conch_Llama3_Patch_VQA\
        --llm_name meta-llama/Meta-Llama-3-8B-Instruct --clip_name conch --data_cache_dir ~/.cache\
        --dataset_name_list CNX-PathLLM/Pathinstruct,CNX-PathLLM/TextbookQAPair,CNX-PathLLM/MultiConversation,CNX-PathLLM/YoutubeInstruct\
        # --data_cache_dir /bask/projects/p/phwq4930-gbm/Zeyu/PathVLM/.cache

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

