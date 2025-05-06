# Test BERT Roberta, POLITICS and BART(fine-tuned)
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime

# Load test set
df = pd.read_csv("test.csv")
texts = df["text"].tolist()
labels = df["label"].tolist()

# Load fine-tuned model
model_path = "D:\Liuyuxuan\MUC\Slp\Dataset and NLP\\final\\fine-tuned_RoBERTa\Model_FT_RoBERTa"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
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
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
accuracy = accuracy_score(labels, preds)

# Print results
print("\n=== Fine-tuned Model Evaluation on test.csv ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# Optional: save to log
with open("metrics_log_fine tuned.txt", "a") as f:
    f.write(f"\n--- Fine-tuned Evaluation @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    f.write(f"Model: fine_tuned_model\n")
    f.write(f"Dataset: test.csv\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1 Score:  {f1:.4f}\n")
