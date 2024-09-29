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

accelerate launch --config_file=./accelerate_configs/deepspeed_zero2.yaml run_qformer_instruct.py --max_steps 200_000\
        --gpu 2 --train_batch_size 24 --eval_batch_size 2 --max_seq_length 256 --resume_from_checkpoint False \
        --output_dir ./output/Conch_Bert_Llama3.1_Stage12_freeze\
        --llm_name meta-llama/Meta-Llama-3.1-8B-Instruct --clip_name conch \
        --bert_name microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext\
        --dataset_name_list CNX-PathLLM/Pathinstruct,CNX-PathLLM/TextbookQAPair,CNX-PathLLM/YoutubeInstruct\
        --data_cache_dir /bask/projects/p/phwq4930-renal-canc/Zeyu/PathVLM/.cache\
        --ckpt /bask/projects/p/phwq4930-gbm/Zeyu/PathVLM/source/PathLLM/output/Conch_Bert_PatchPre/ckpt19500.bin\
        --llm_requires_grad False # --use_peft True --peft_lora_r 8 


#CNX-PathLLM/TextbookQAPair,CNX-PathLLM/MultiConversation,CNX-PathLLM/YoutubeInstruct
