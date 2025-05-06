#1和0的结果和其他结果是反过来的，所以需要互换1和0
#
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ========== Configuration ==========
MODEL_NAME = "launch/POLITICS"  # or "bert-base-uncased"
CSV_PATH = "D:\Liuyuxuan\MUC\Slp\Dataset and NLP\\final\\task\kmeans-data.csv"

OUTPUT_CSV = "ALL_clustered_texts.csv"
SHOW_TSNE = True  # set to False if t-SNE is too slow
NUM_SAMPLES_PER_CLUSTER = 5
DEVICE = "cpu"

# ========== Load Data ==========
df = pd.read_csv(CSV_PATH, encoding='gb18030')  # 👈 修复了编码问题
texts = df["text"].tolist()

# ========== Load Model ==========
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

# ========== Encode Texts ==========
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # CLS token

print("Encoding texts...")
embeddings = [get_embedding(text) for text in texts]

# ========== K-Means Clustering ==========
print("Clustering...")
kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)
df["cluster"] = cluster_labels

# ========== Save Results ==========
df.to_csv(OUTPUT_CSV, index=False)
print(f"Clustered results saved to {OUTPUT_CSV}")

# ========== Show Example Texts per Cluster ==========
print("\nSample texts from each cluster:")
for c in sorted(df["cluster"].unique()):
    print(f"\n--- Cluster {c} ---")
    cluster_df = df[df["cluster"] == c]
    sample_n = min(NUM_SAMPLES_PER_CLUSTER, len(cluster_df))  # 👈 防止报错
    samples = cluster_df.sample(sample_n, random_state=42)
    for i, row in samples.iterrows():
        print(f"• {row['text'][:120]}...")

# ========== Visualize Clusters ==========
reduced_pca = PCA(n_components=2).fit_transform(embeddings)
plt.figure(figsize=(10, 8))
for i, (x, y) in enumerate(reduced_pca):
    plt.scatter(x, y, c="C{}".format(cluster_labels[i]), alpha=0.6)
    plt.text(x + 0.1, y + 0.1, str(df.iloc[i]["id"]), fontsize=8)

plt.title(f"K-Means Clustering ({MODEL_NAME}) via PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()
if SHOW_TSNE:
    print("Running t-SNE (may take a while)...")
    reduced_tsne = TSNE(n_components=2, perplexity=30).fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    for i, (x, y) in enumerate(reduced_tsne):
        plt.scatter(x, y, c="C{}".format(cluster_labels[i]), alpha=0.6)
        plt.text(x + 0.1, y + 0.1, str(df.iloc[i]["id"]), fontsize=8)

    plt.title(f"K-Means Clustering ({MODEL_NAME}) via t-SNE")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.show()
