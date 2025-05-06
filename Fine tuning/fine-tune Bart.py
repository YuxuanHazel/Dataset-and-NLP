import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    BartForSequenceClassification,
    Trainer,
    TrainingArguments
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# === 1. Load data ===
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

# === 2. Load BART tokenizer and model ===
model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BartForSequenceClassification.from_pretrained(model_name, num_labels=2)

# === 3. Tokenization ===
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

train_ds = train_ds.map(tokenize_function, batched=True)
val_ds = val_ds.map(tokenize_function, batched=True)
train_ds = train_ds.remove_columns(["text"])
val_ds = val_ds.remove_columns(["text"])
train_ds.set_format("torch")
val_ds.set_format("torch")

# === 4. Metrics ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# === 5. Training arguments ===
training_args = TrainingArguments(
    output_dir="./results_bart",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# === 6. Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# === 7. Train ===
trainer.train()

# === 8. Save fine-tuned model ===
trainer.save_model("fine_tuned_bart_model")
