import pandas as pd
from sklearn.model_selection import train_test_split

# Load the original twin-view dataset
df = pd.read_csv("D:\\Liuyuxuan\\MUC\\Slp\\Dataset and NLP\\final\\twinviews-13k.csv")

# Create left-leaning samples (label = 0)
left_df = pd.DataFrame({
    "text": df["l"],
    "label": 0
})

# Create right-leaning samples (label = 1)
right_df = pd.DataFrame({
    "text": df["r"],
    "label": 1
})

# Combine into a single dataset
combined_df = pd.concat([left_df, right_df], ignore_index=True)

# Save full dataset (optional, for reference)
combined_df.to_csv("dataset.csv", index=False)

# Split: 80% train, 10% val, 10% test
train_df, temp_df = train_test_split(combined_df, test_size=0.2, random_state=42, stratify=combined_df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

# Save each split
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Dataset conversion and 80/10/10 split complete.")
print("Files saved as: train.csv, val.csv, test.csv")
