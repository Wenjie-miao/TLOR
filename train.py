import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_path = "your_model_path"  
dataset_path = "your_dataset_path" 
output_dir = "your/output/file/path"  
log_dir = "./logs"                  

use_qlora = True  
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           
    bnb_4bit_compute_dtype=torch.bfloat16, 
    bnb_4bit_use_double_quant=True,      
)

lora_config = LoraConfig(
    r=16,                              
    lora_alpha=32,                      
    target_modules=[                    
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        

    ],
    lora_dropout=0.05,                  
    bias="none",                        
    task_type="CAUSAL_LM",              
)


training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,      
    gradient_accumulation_steps=4,      
    num_train_epochs=1,                 
    learning_rate=1e-4,                 
    logging_dir=log_dir,
    logging_steps=50,                   
    save_steps=200,                     
    save_total_limit=2,                 
    lr_scheduler_type="cosine",         
    warmup_ratio=0.03,                  
    optim="paged_adamw_8bit" if use_qlora else "adamw_torch", 
    fp16=True if not torch.cuda.is_bf16_supported() else False, 
    bf16=torch.cuda.is_bf16_supported(), 
    report_to="tensorboard",            
    gradient_checkpointing=True, 
    ddp_find_unused_parameters=False,       

)


max_seq_length = 2048 


tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

if tokenizer.pad_token is None:
    
    tokenizer.pad_token = tokenizer.eos_token
    print("Pad token set to EOS token:", tokenizer.pad_token)


tokenizer.padding_side = 'left'


print("Loading dataset...")
dataset = load_dataset("json", data_files=dataset_path, split="train") # 加载 json 数据


def format_and_tokenize(example):
    
    try:
        
        messages = [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]}
        ]
        
        full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False # 因为我们有 output，这是训练数据
        )
    except Exception as e:
        # 如果 apply_chat_template 不可用或出错，回退到简单模板
        print(f"Warning: Failed to apply chat template ({e}). Using basic template.")
        full_prompt = f"### Instruction:\n{example['input']}\n\n### Response:\n{example['output']}{tokenizer.eos_token}" # 手动添加 eos

    
    tokenized_output = tokenizer(
        full_prompt,
        truncation=True,
        max_length=max_seq_length,
        padding=False, 
    )
    return tokenized_output


print("Preprocessing dataset...")
tokenized_dataset = dataset.map(
    format_and_tokenize,
    remove_columns=list(dataset.column_names) 
)
print(f"Dataset size after tokenization: {len(tokenized_dataset)}")
print("Example tokenized data:", tokenized_dataset[0])



print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config if use_qlora else None,

    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, # 匹配计算类型
)

if use_qlora:
    print("Preparing model for QLoRA training...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

print("Applying LoRA/QLoRA...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # 打印可训练参数数量，确认 LoRA 生效

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)


print("Starting training...")
train_result = trainer.train()


print("Saving final LoRA adapters...")

model.save_pretrained(output_dir)

tokenizer.save_pretrained(output_dir)

trainer.save_state()

metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

print("\n--- Fine-tuning completed! ---")
print(f"LoRA adapters saved to: {output_dir}")
print("You can now load the base model and apply these adapters for inference.")

