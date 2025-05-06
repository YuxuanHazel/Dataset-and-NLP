#"D:\Liuyuxuan\MUC\Slp\Dataset and NLP\final\fine_tuned_POLITICS\Model_FT_POLITICS"
#"D:\Liuyuxuan\MUC\Slp\Dataset and NLP\final\fine-tuned_RoBERTa\Model_FT_RoBERTa"
# csv_path = r"D:\Liuyuxuan\MUC\Slp\Dataset and NLP\final\task\ALL_clustered_texts.csv"
# model_path = r"D:\Liuyuxuan\MUC\Slp\Dataset and NLP\final\fine_tuned_Bert\Model_FT_Bert"
# output_dir = r"D:\Liuyuxuan\MUC\Slp\Dataset and NLP\final\task\SHAP_output\BERT"
import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ==== changeable ====
csv_path = r"D:\Liuyuxuan\MUC\Slp\Dataset and NLP\final\task\ALL_clustered_texts.csv"
model_path = r"D:\Liuyuxuan\MUC\Slp\Dataset and NLP\final\fine-tuned_RoBERTa\Model_FT_RoBERTa"
output_dir = r"D:\Liuyuxuan\MUC\Slp\Dataset and NLP\final\task\SHAP_output\RoBERTa"
model_name = "RoBERTa-GPT"

os.makedirs(output_dir, exist_ok=True)

# ==== 加载数据，只取 id = 1 ~ 15 ====
df = pd.read_csv(csv_path)
df = df[df["id"].isin(range(31,46))].dropna(subset=["text"])
texts = df["text"].tolist()


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True, device=-1)


predictions = classifier(texts)
predicted_labels = [max(p, key=lambda x: x['score'])['label'] for p in predictions]


explainer = shap.Explainer(classifier, masker=shap.maskers.Text(tokenizer))
shap_values = explainer(texts)


left_scores = defaultdict(list)
right_scores = defaultdict(list)

for i, val in enumerate(shap_values):
    label = predicted_labels[i]  # 'LABEL_0' or 'LABEL_1'
    target_dict = left_scores if label == 'LABEL_0' else right_scores

    for word, score in zip(val.data, val.values):
        if isinstance(word, str):

            if isinstance(score, (list, tuple, np.ndarray)):
                score_scalar = float(np.mean(score))
            else:
                score_scalar = float(score)
            target_dict[word].append(score_scalar)


avg_left = {w: float(np.mean(v)) for w, v in left_scores.items() if v}
avg_right = {w: float(np.mean(v)) for w, v in right_scores.items() if v}


WordCloud(width=600, height=300, background_color="white").generate_from_frequencies(avg_left).to_file(
    os.path.join(output_dir, f"{model_name}_left_wordcloud.png"))
WordCloud(width=600, height=300, background_color="white").generate_from_frequencies(avg_right).to_file(
    os.path.join(output_dir, f"{model_name}_right_wordcloud.png"))


def plot_top_bar(avg_dict, title, filename, color):
    top = sorted(avg_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    if not top:
        print(f"⚠️ No data to plot for {title}.")
        return
    words, values = zip(*top)
    plt.figure()
    plt.barh(words, values, color=color)
    plt.xlabel("Average SHAP Value")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

plot_top_bar(avg_left, "Top 10 Left-Leaning Words", f"{model_name}_top_left_bar.png", "salmon")
plot_top_bar(avg_right, "Top 10 Right-Leaning Words", f"{model_name}_top_right_bar.png", "skyblue")

print("Done")


