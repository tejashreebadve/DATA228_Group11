import pandas as pd
import pyarrow.parquet as pq
from transformers import pipeline
import torch

# ✅ Check GPU
device = 0 if torch.cuda.is_available() else -1
print(f"Using {'GPU' if device == 0 else 'CPU'} for inference")

# ✅ Load Parquet file directly with pandas
df = pd.read_parquet(r"C:\semester_2\Data_228\project\steam_reviews_cleaned\steam_reviews_cleaned.parquet")

# ✅ Load Hugging Face sentiment classifier
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)

# ✅ Batch process
batch_size = 100_000
total_rows = len(df)
num_batches = (total_rows // batch_size) + 1

results = []

for i in range(num_batches):
    start = i * batch_size
    end = min(start + batch_size, total_rows)
    print(f"Processing batch {i+1}/{num_batches} — Rows {start} to {end}")

    batch_texts = df["cleaned_review"].iloc[start:end].tolist()

    outputs = classifier(batch_texts, batch_size=32, truncation=True)

    labels = [o["label"] for o in outputs]
    scores = [o["score"] for o in outputs]

    batch_result = pd.DataFrame({
        "cleaned_review": df["cleaned_review"].iloc[start:end].values,
        "sentiment_label": labels,
        "sentiment_score": scores
    })

    results.append(batch_result)

# ✅ Concatenate all batches
final_df = pd.concat(results, ignore_index=True)

# ✅ Save to Parquet
final_df.to_parquet(r"C:\semester_2\Data_228\project\steam_sentiment_final_batched.parquet")

print("Sentiment analysis complete. Saved to Parquet.")
