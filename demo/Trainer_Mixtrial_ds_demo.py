
# accelerate launch --config_file=/raid/hpc/hekai/WorkShop/My_project/PathLLM_new/accelerate_configs/deepspeed_zero2.yaml  Trainer_Mixtrial_demo.py 

import os
os.environ["WANDB_MODE"] = "offline"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
from torch import nn
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModel, AutoModelForCausalLM,  BitsAndBytesConfig, HfArgumentParser, Trainer, PreTrainedModel
from accelerate import Accelerator
from datasets import load_dataset
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="mistralai/Mistral-7B-Instruct-v0.2", metadata={"help": "the model name，  meta-llama/Llama-2-7b-chat-hf "})
    dataset_name: Optional[str] = field(default="stingning/ultrachat", metadata={"help": "the dataset name"})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=2.0e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=1, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(default=8, metadata={"help": "the number of gradient accumulation steps"})
    
    evaluation_strategy: Optional[str] = field(default="steps", metadata={"help": "epoch, step"})
    eval_steps: Optional[int] = field(default=1000, metadata={"help": "the number of gradient accumulation steps"})

    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
   
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=5, metadata={"help": "the number of logging steps"})
    token: Optional[bool] = field(default="True", metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(default=1000, metadata={"help": "Number of updates steps before two checkpoint saves"})
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default="mistral-7b-finetuned-ultrachat", metadata={"help": "The name of the model on HF Hub"}
)

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]




class MyCustomModel(nn.Module):
    def __init__(self, script_args, num_labels):
        super(MyCustomModel, self).__init__()
        self.num_labels = num_labels
        self.pretrained_model = AutoModelForCausalLM.from_pretrained(script_args.model_name,
                                                        quantization_config=quantization_config,
                                                        device_map=device_map,
                                                        trust_remote_code=script_args.trust_remote_code,
                                                        torch_dtype=torch_dtype,
                                                        token=script_args.token)
        self.classifier = nn.Linear(self.pretrained_model.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.pretrained_model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state[:,0,:]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits} if loss is not None else logits


dataset = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    # Tokenize the inputs (pair of sentences)
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding=True, max_length=10)




from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

small_train_dataset = dataset["train"].shuffle(seed=42).select(range(500)) # 选择前500个样本
small_train_dataset = small_train_dataset.map(preprocess_function, batched=True)


if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
    )
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}
    torch_dtype = torch.bfloat16
else:
    device_map = None
    quantization_config = None
    torch_dtype = None


model = MyCustomModel(script_args, num_labels=2)



training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    # gradient_checkpointing=True,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    evaluation_strategy=script_args.evaluation_strategy,
    eval_steps=script_args.eval_steps,
    logging_first_step=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    data_collator=data_collator,  
    compute_metrics=None, 
)


trainer.train()
# model.save_pretrained("./my_custom_model")