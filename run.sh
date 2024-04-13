#!/bin/bash
#SBATCH --account=phwq4930-gbm
#SBATCH --qos=epsrc
#SBATCH --gpus-per-task 1
#SBATCH --tasks-per-node 1
#SBATCH --nodes 1
#SBATCH --time 1:0:0
#SBATCH --mem 256G
#SBATCH -o ./logs/slurm.%N.%j.out # STDOUT
#SBATCH -e ./logs/slurm.%N.%j.err # STDERR

# module purge
# module load baskerville
# module load GCCcore/10.2.0 Python/3.8.6
# module load CUDA/11.7.0
# module load cuDNN/8.4.1.50-CUDA-11.7.0

module purge
module load bask-apps/live
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0

# source /bask/projects/p/phwq4930-gbm/Zeyu/pyvenv/medfla2/bin/activate
source /bask/projects/p/phwq4930-gbm/Zeyu/pyvenv/pathllmGZY/bin/activate
export WANDB_MODE=online

accelerate launch --config_file=/bask/homes/a/asiw9691/PathVLM/source/PathLLM/accelerate_configs/deepspeed_zero2.yaml run.py  --gpu 2 --train_batch_size 8   --eval_batch_size 8 --max_seq_length 256

# accelerate launch --config_file=/raid/hpc/hekai/WorkShop/My_project/PathLLM_new/accelerate_configs/deepspeed_zero2.yaml  run.py --gpu 6 --train_batch_size 16   --eval_batch_size 16 --max_seq_length 256

# accelerate launch --config_file=/raid/hpc/hekai/WorkShop/My_project/PathLLM_new/accelerate_configs/deepspeed_zero3.yaml  run.py
