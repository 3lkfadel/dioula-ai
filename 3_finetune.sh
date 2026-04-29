#!/bin/bash

# ─── Llama 3.1 70B — AWS (GPU A100/H100 recommandé) ───────────────────────
# Télécharger le modèle avant de lancer :
#   huggingface-cli download meta-llama/Meta-Llama-3.1-70B-Instruct \
#     --local-dir ./model/llama-3.1-70b

MODEL_PATH="./model/llama-3.1-70b"
ADAPTER_PATH="./model/adapters"

echo "Fine-tuning Dioula-AI — Llama 3.1 70B — dataset propre 38 027 paires"
echo "======================================="

mkdir -p "$ADAPTER_PATH"

# Sur AWS : utiliser torchrun ou accelerate selon l'infra disponible
# Ici on suppose un environnement HuggingFace + PEFT + bitsandbytes (QLoRA)

python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import torch

MODEL_ID = '${MODEL_PATH}'
DATA_PATH = '.'

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    load_in_4bit=True,           # QLoRA 4-bit
    torch_dtype=torch.float16,
    device_map='auto',
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir='${ADAPTER_PATH}',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    eval_strategy='steps',
    eval_steps=200,
    warmup_ratio=0.05,
    lr_scheduler_type='cosine',
    report_to='none',
)

dataset = load_dataset('json', data_files={
    'train': f'{DATA_PATH}/train.jsonl',
    'validation': f'{DATA_PATH}/valid.jsonl',
})

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=tokenizer,
    dataset_text_field='prompt',
    max_seq_length=512,
)

trainer.train()
trainer.save_model('${ADAPTER_PATH}/final')
print('Fine-tuning termine !')
"

echo "Fine-tuning termine !"
