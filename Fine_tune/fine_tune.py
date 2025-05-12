import json
from transformers import AutoModelForCausalLM, AutoTokenizer , TrainingArguments , Trainer
from peft import LoraConfig , get_peft_model
from datasets import load_dataset
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


with open('question_answer.json', 'r') as f:
    qa_pairs = json.load(f)

with open('train.jsonl', 'w') as f:
    for qa in qa_pairs:
        prompt = f"### Instruction:\nYou are an expert in Algorand Blockchain development. Answer the following question concisely and accurately.\n\n### Question:\n{qa['question']}\n\n### Answer:\n{qa['answer']}"
        entry = {"text": prompt}
        f.write(json.dumps(entry) + '\n')




model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


if tokenizer.pad_token is None:
    tokenizer.pad_token = '[PAD]'



lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

dataset = load_dataset('json', data_files={'train': 'train.jsonl'})

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

training_args = TrainingArguments(
    output_dir='./fine_tuned_llama',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=None,
    data_collator=lambda data: {
        'input_ids': torch.stack([f['input_ids'] for f in data]),
        'attention_mask': torch.stack([f['attention_mask'] for f in data]),
        'labels': torch.stack([f['input_ids'] for f in data]),
    }
)

trainer.train()
model.save_pretrained('./fine_tuned_llama')
tokenizer.save_pretrained('./fine_tuned_llama')
