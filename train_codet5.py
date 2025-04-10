import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

model_name = "Salesforce/codet5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def load_processed(file_path):
    df = pd.read_csv(file_path)
    return Dataset.from_pandas(df)

dataset = {
    'train': load_processed("processed_data/train.csv"),
    'validation': load_processed("processed_data/valid.csv")
}

def tokenize_function(examples):
    model_inputs = tokenizer(examples["input"], max_length=512, padding="max_length", truncation=True)
    labels = tokenizer(examples["target"], max_length=64, padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = {
    'train': dataset['train'].map(tokenize_function, batched=True),
    'validation': dataset['validation'].map(tokenize_function, batched=True)
}

output_dir = "codet5_finetuned"
os.makedirs(output_dir, exist_ok=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

trainer.train()
trainer.save_model(output_dir)
print(f"Model fine-tuning complete and saved to '{output_dir}/'")
