


from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

model_name_or_path = "/raid/hpc/hekai/WorkShop/My_project/LLM_models/llama2/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

peft_config = LoraConfig( 
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM, 
)

model = get_peft_model(model, peft_config)