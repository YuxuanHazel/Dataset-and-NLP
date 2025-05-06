import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datetime import datetime
from collections import Counter

# Load test set
df = pd.read_csv("test.csv")
texts = df["text"].tolist()
labels = df["label"].tolist()

# List of base models to test
model_list = [
    "bert-base-uncased",
    "roberta-base"
]

for model_name in model_list:
    print(f"\n=== Zero-shot Evaluation: {model_name} ===")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.eval()

    preds = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        preds.append(pred)

    # Analyze prediction distribution
    pred_counts = Counter(preds)
    print("Prediction distribution:", pred_counts)

    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    accuracy = accuracy_score(labels, preds)

    # Print metrics
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Log results
    with open("metrics_log.txt", "a") as f: # a 表示会写在后面
        f.write(f"\n--- Zero-shot Evaluation @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: test.csv\n")
        f.write(f"Prediction distribution: {dict(pred_counts)}\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
