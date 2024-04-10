

nohup accelerate launch --config_file=/raid/hpc/hekai/WorkShop/My_project/PathLLM_new/accelerate_configs/deepspeed_zero2.yaml  run.py  >.log 2>&1 &



accelerate launch --config_file=/raid/hpc/hekai/WorkShop/My_project/PathLLM_new/accelerate_configs/deepspeed_zero2.yaml  run.py
accelerate launch --config_file=/raid/hpc/hekai/WorkShop/My_project/PathLLM_new/accelerate_configs/deepspeed_zero3.yaml  run.py


# train_batch_size: Optional[int] = field(default=40, metadata={"help": "the batch size"})
# eval_batch_size: Optional[int] = field(default=48, metadata={"help": "the batch size"})
# max_seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})