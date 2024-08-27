#!/bin/bash
#SBATCH --account=phwq4930-gbm
#SBATCH --qos=epsrc
#SBATCH --gpus-per-task 1
#SBATCH --tasks-per-node 1
#SBATCH --nodes 1
#SBATCH --time 4:0:0
#SBATCH --mem 256G
#SBATCH --constraint=a100_40
#SBATCH -o ./logs/slurm.%N.%j.out # STDOUT
#SBATCH -e ./logs/slurm.%N.%j.err # STDERR

module purge
module load bask-apps/live
module load Java/11
module load GCCcore/11.3.0 Python/3.10.4
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

source /bask/projects/p/phwq4930-renal-canc/Zeyu/pyvenv/pathllmGZY/bin/activate

python test_patch_instruct.py --ckpt_path /bask/homes/a/asiw9691/PathVLM/source/PathLLM/output/Conch_Llama3_Patch_VQA/ckpt10500.bin