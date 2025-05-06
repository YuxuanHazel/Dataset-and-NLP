# Test Bart评估模型
import pandas as pd
import torch
from transformers import AutoTokenizer, BartForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load test data
df = pd.read_csv("D:\Liuyuxuan\MUC\Slp\Dataset and NLP\\final\\test.csv")
texts = df["text"].tolist()
labels = df["label"].tolist()

# Load untrained BART for classification
model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BartForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.eval()

# Predict
preds = []
for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    preds.append(pred)

# Metrics
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
accuracy = accuracy_score(labels, preds)

print("\n=== Zero-shot BART (No Fine-tuning) ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

from collections import Counter
from datetime import datetime

# Count prediction distribution
from collections import Counter
pred_counts = Counter(preds)

# Append results to shared log file
with open("metrics_log.txt", "a") as f:  # "a" means append
    f.write(f"\n--- Zero-shot Evaluation @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    f.write(f"Model: {model_name}\n")
    f.write(f"Dataset: test.csv\n")
    f.write(f"Prediction distribution: {dict(pred_counts)}\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1 Score:  {f1:.4f}\n")

