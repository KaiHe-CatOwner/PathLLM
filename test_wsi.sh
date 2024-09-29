#!/bin/bash
#SBATCH --account=phwq4930-gbm
#SBATCH --qos=epsrc
#SBATCH --gpus-per-task 1
#SBATCH --tasks-per-node 1
#SBATCH --nodes 1
#SBATCH --time 24:0:0
#SBATCH --mem 256G
#SBATCH --constraint=a100_40
#SBATCH -o ./logs/slurm.%N.%j.out # STDOUT
#SBATCH -e ./logs/slurm.%N.%j.err # STDERR

# module purge
# module load baskerville
# module load GCCcore/10.2.0 Python/3.8.6
# module load CUDA/11.7.0
# module load cuDNN/8.4.1.50-CUDA-11.7.0

module purge
module load bask-apps/live
module load GCCcore/11.3.0 Python/3.10.4
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0
source /bask/projects/p/phwq4930-renal-canc/Zeyu/pyvenv/pathllmGZY/bin/activate

python test_wsi.py --max_seq_length 256 --batch_size 4 --llm_name meta-llama/Meta-Llama-3.1-8B-Instruct\
                    --shuffle True --data_cache_dir /bask/projects/p/phwq4930-renal-canc/Zeyu/PathVLM/.cache\
                    --dataset_name_list CNX-PathLLM/GTEx-WSI-Description,CNX-PathLLM/TCGA-WSI-Description-4onew,CNX-PathLLM/TCGA-WSI-Description-4omini\
                    --eval_sample_size -1 --n_heads 32,16,8 --agg_strategy abmil --embed_dim 512\
                    --fea_root /bask/homes/a/asiw9691/PathVLM/WSI_Dataset/Conch \
                    --ckpt_path /bask/homes/a/asiw9691/PathVLM/source/PathLLM/output/WSI_ConchLlama3.1_abmil_Stage1/ckpt20000.bin
