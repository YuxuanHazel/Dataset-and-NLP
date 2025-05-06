import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# ====== Config ======
input_file = "D:\Liuyuxuan\MUC\Slp\Dataset and NLP\\final\\task\ALL_clustered_texts.csv"
model_path = "D:\Liuyuxuan\MUC\Slp\Dataset and NLP\\final\\fine_tuned_POLITICS\Model_FT_POLITICS"

# ====== Load Data ======
df = pd.read_csv(input_file, encoding="gb18030")

texts = df["text"].tolist()

# ====== Load Model ======
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# ====== Predict ======
predicted_labels = []
probabilities = []

for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    predicted_labels.append(pred)
    probabilities.append(round(confidence, 4))  

# ====== Save Predictions ======
df["predicted_label"] = predicted_labels
df["probability"] = probabilities


input_name = os.path.splitext(os.path.basename(input_file))[0]
output_file = f"{input_name}_predictions_POLITICS.csv"
df.to_csv(output_file, index=False)

print(f"Predictions with probabilities saved to {output_file}")
