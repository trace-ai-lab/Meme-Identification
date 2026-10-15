import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    set_seed
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODEL_NAME = "xlm-roberta-large" 
MAX_LENGTH = 128                
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

SEEDS = [0, 1, 2, 3, 4]

print(f"loading {MODEL_NAME} tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def load_and_prepare_data(csv_file):
    df = pd.read_csv(csv_file)
    df['template'] = df['template'].fillna('N/A')
    df['idea_text'] = df['idea_text'].fillna('')
    
    df['text'] = "Template: " + df['template'] + " | Idea: " + df['idea_text']
    df['label'] = df['label'].astype(int)
    
    dataset = Dataset.from_pandas(df[['text', 'label']])
    return dataset

def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_LENGTH
    )


print("loading and preparing dataset...")
train_dataset = load_and_prepare_data("train_set.csv")
dev_dataset = load_and_prepare_data("dev_set.csv")
test_dataset = load_and_prepare_data("test_set.csv")

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_dev = dev_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, zero_division=0)
    rec = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

metric_keys = ['eval_accuracy', 'eval_precision', 'eval_recall', 'eval_f1']
all_results = {
    'base_dev': {k: [] for k in metric_keys},
    'base_test': {k: [] for k in metric_keys},
    'ft_dev': {k: [] for k in metric_keys},
    'ft_test': {k: [] for k in metric_keys}
}

for seed in SEEDS:
    print("\n" + "="*60)
    print(f" 开始运行 Seed: {seed}")
    print("="*60)
    
    set_seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=f"./xlmr-meme-classifier_seed_{seed}",
        eval_strategy="epoch",        
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,      
        metric_for_best_model="f1",        
        logging_dir=f'./logs_seed_{seed}',
        logging_steps=10,
        save_total_limit=1 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )


    print(f"[{seed}] evaluating baseline model...")
    base_dev = trainer.evaluate(tokenized_dev)
    base_test = trainer.evaluate(tokenized_test)
    
    for k in metric_keys:
        all_results['base_dev'][k].append(base_dev[k])
        all_results['base_test'][k].append(base_test[k])


    print(f"[{seed}] fine-tuning model...")
    trainer.train()


    print(f"[{seed}] evaluating fine-tuned model...")
    ft_dev = trainer.evaluate(tokenized_dev)
    ft_test = trainer.evaluate(tokenized_test)
    
    for k in metric_keys:
        all_results['ft_dev'][k].append(ft_dev[k])
        all_results['ft_test'][k].append(ft_test[k])


print("\n\n" + "#"*60)
print("Final 5 Runs Results (Mean ± SE)")
print("#"*60)


def print_mean_se_metrics(results_dict, set_name):
    print(f"\n=== {set_name} ===")
    for k in metric_keys:
        data = results_dict[k]
        mean_val = np.mean(data)
        se_val = np.std(data, ddof=1) / np.sqrt(len(data))
        
        metric_name = k.replace('eval_', '').capitalize()
        print(f"  {metric_name:<10}: {mean_val:.4f} ± {se_val:.4f}")

print_mean_se_metrics(all_results['base_dev'], "Dev Set - Baseline")
print_mean_se_metrics(all_results['ft_dev'],   "Dev Set - Fine-tuned")

print_mean_se_metrics(all_results['base_test'], "Test Set - Baseline")
print_mean_se_metrics(all_results['ft_test'],   "Test Set - Fine-tuned")

