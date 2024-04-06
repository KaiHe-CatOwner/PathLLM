import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
from torch import nn
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# 自定义模型，继承自nn.Module或者transformers提供的预训练模型类
class MyCustomModel(nn.Module):
    def __init__(self, num_labels):
        super(MyCustomModel, self).__init__()
        self.num_labels = num_labels
        self.pretrained_model = AutoModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(self.pretrained_model.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.pretrained_model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[1]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits} if loss is not None else logits

# 加载数据集并预处理
dataset = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    # Tokenize the inputs (pair of sentences)
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding=True)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

small_train_dataset = dataset["train"].shuffle(seed=42).select(range(500)) # 选择前500个样本
small_train_dataset = small_train_dataset.map(preprocess_function, batched=True)

for i in small_train_dataset:
    print(i)


# 自定义模型实例化
model = MyCustomModel(num_labels=2).to("cuda")

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    data_collator=data_collator,  
    compute_metrics=None, # 如果需要可以添加计算指标的函数
)

# 训练模型
trainer.train()

# 保存模型
model.save_pretrained("./my_custom_model")