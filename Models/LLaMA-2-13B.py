from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoConfig
)
from datasets import Dataset
import pandas as pd
import numpy as np
import torch
import gc
import evaluate
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login

os.environ['TRANSFORMERS_CACHE'] = '/scratch/nzafarm/cache/'
os.environ['HF_HOME'] = '/scratch/nzafarm/cache/'
os.environ['HF_TOKEN'] = 'YOUR_HF_TOKEN_HERE'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

df = pd.read_csv('Charleston_dataset.csv')
df = df.dropna(subset=['Remark'])

exclude_events = []
df = df[~df['Event Type'].isin(exclude_events)]

event_type_mapping = {
    "FLASH FLOOD": "FLOOD", "COASTAL FLOOD": "FLOOD", "STORM SURGE": "FLOOD",
    "HIGH ASTR TIDES": "FLOOD", "HEAVY RAIN": "FLOOD",
    "TROPICAL STORM": "CYCLONIC", "HURRICANE": "CYCLONIC", "TROPICAL CYCLONE": "CYCLONIC",
    "TSTM WIND": "THUNDERSTORM", "TSTM WND DMG": "THUNDERSTORM", "TSTM WND GST": "THUNDERSTORM",
    "MARINE TSTM WIND": "THUNDERSTORM"
}
df['label'] = df['Event Type'].replace(event_type_mapping)
event_type_mapping_numeric = {event: i for i, event in enumerate(df['label'].unique())}
df['label'] = df['label'].map(event_type_mapping_numeric)

df = df[['Remark', 'label']].rename(columns={'Remark': 'text'})
dataset = Dataset.from_pandas(df[['text', 'label']])

import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

model_name = 'meta-llama/Llama-2-13b-hf'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.shuffle(seed=42)

full_train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
full_train_dataset = full_train_test_split['train']
full_test_dataset = full_train_test_split['test']

num_labels = len(event_type_mapping_numeric)
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    overall_accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    label_names = {v: k for k, v in event_type_mapping_numeric.items()}
    metrics = {'eval_overall_accuracy': overall_accuracy}
    for i in range(len(precision)):
        class_name = label_names[i]
        metrics[f'eval_precision_{class_name}'] = precision[i]
        metrics[f'eval_recall_{class_name}'] = recall[i]
        metrics[f'eval_f1_{class_name}'] = f1[i]
        metrics[f'eval_support_{class_name}'] = support[i]
    return metrics

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").to(model.device)
        outputs = model(**inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs
        loss = torch.nn.functional.cross_entropy(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

data_sizes = [i / 10 for i in range(1, 11)]
results = []

for data_size in data_sizes:
    subset_size = int(data_size * len(full_train_dataset))
    print(f"\nTraining with {subset_size} samples ({data_size * 100}%)")
    train_subset = full_train_dataset.select(range(subset_size))
    test_subset_size = int(data_size * len(full_test_dataset))
    test_subset = full_test_dataset.select(range(test_subset_size))

    try:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except NameError:
        pass

    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    config.use_cache = False

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        device_map='auto',
        load_in_8bit=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=f'./results_{int(data_size * 100)}',
        evaluation_strategy="epoch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        save_strategy="epoch",
        logging_dir='./logs',
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=1e-4,
        max_grad_norm=0.3,
        logging_steps=10
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=test_subset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_results = trainer.evaluate()

    result = {
        'data_size': data_size,
        'eval_overall_accuracy': eval_results['eval_overall_accuracy']
    }
    for key in eval_results:
        if key.startswith('eval_') and key != 'eval_overall_accuracy':
            result[key] = eval_results[key]
    results.append(result)

    with open(f'llama_lora_results_{int(data_size * 100)}.txt', 'w') as f:
        f.write(f"Evaluation Results for data size {int(data_size * 100)}%:\n")
        for key, value in eval_results.items():
            f.write(f"{key}: {value}\n")

    del model
    torch.cuda.empty_cache()
    gc.collect()

results_df = pd.DataFrame(results)
results_df.to_csv('llama-2-7b.csv', index=False)
print(results_df)
