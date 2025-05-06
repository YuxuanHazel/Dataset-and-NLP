import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datetime import datetime

# Load test data (10%)
df = pd.read_csv("test.csv")
texts = df["text"].tolist()
labels = df["label"].tolist()

# Choose the model (zero-shot test on POLITICS)
model_name = "launch/POLITICS"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.eval()

# Run predictions
preds = []
for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    preds.append(pred)

# Evaluate
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
accuracy = accuracy_score(labels, preds)

# Print results
print("\n=== Zero-shot Evaluation on test.csv ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# Save results to metrics log
with open("metrics_log.txt", "a") as f:
    f.write(f"\n--- Zero-shot Evaluation @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    f.write(f"Model: POLITICS (launch/POLITICS)\n")
    f.write(f"Dataset: test.csv\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1 Score:  {f1:.4f}\n")
