import pandas as pd
from transformers import pipeline
import torch

# ✅ Check GPU
device = 0 if torch.cuda.is_available() else -1
print(f"✅ Using {'GPU' if device == 0 else 'CPU'} for inference")

# ✅ Load full dataframe (with app_id, app_name, review_id, cleaned_review)
df = pd.read_parquet(r"C:\semester_2\Data_228\project\steam_reviews_cleaned\steam_reviews_cleaned.parquet")
print(f"✅ Loaded {len(df)} rows")

# ✅ Load sentiment pipeline
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)

# ✅ Parameters
pipeline_batch_size = 32  # control inside pipeline
external_batch_size = 50000  # how many rows to process per outer loop
total_rows = len(df)
num_batches = (total_rows // external_batch_size) + 1

results = []

for i in range(num_batches):
    start = i * external_batch_size
    end = min(start + external_batch_size, total_rows)
    print(f"\n🚀 Processing batch {i+1}/{num_batches}: rows {start}-{end}")

    batch_df = df.iloc[start:end]
    texts = batch_df["cleaned_review"].tolist()

    # ✅ Run pipeline on this batch
    outputs = classifier(texts, batch_size=pipeline_batch_size, truncation=True)

    # ✅ Build result DataFrame with original columns
    batch_result = pd.DataFrame({
        "app_id": batch_df["app_id"].values,
        "app_name": batch_df["app_name"].values,
        "review_id": batch_df["review_id"].values,
        "cleaned_review": batch_df["cleaned_review"].values,
        "sentiment_label": [o["label"] for o in outputs],
        "sentiment_score": [o["score"] for o in outputs]
    })

    results.append(batch_result)

# ✅ Concatenate everything
final_df = pd.concat(results, ignore_index=True)

# ✅ Save to Parquet
final_df.to_parquet(r"C:\semester_2\Data_228\project\steam_sentiment_final_batched.parquet")
print("✅ Sentiment analysis complete and saved to Parquet.")
