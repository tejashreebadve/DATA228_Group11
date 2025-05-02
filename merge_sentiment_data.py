from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder \
    .appName("MergeSentimentWithCleanedReviews") \
    .getOrCreate()

# 🟢 Load original cleaned reviews
cleaned_df = spark.read.parquet("/user/tejashree/project/data/processed/cleaned_steam_reviews.parquet")

# 🟢 Load sentiment data uploaded by you
sentiment_df = spark.read.parquet("/user/tejashree/project/data/processed/steam_sentiment_final_batched.parquet")

# 🟢 Join on review_id to add sentiment columns
merged_df = cleaned_df.join(
    sentiment_df.select("review_id", "sentiment_label", "sentiment_score"),
    on="review_id",
    how="left"
)

# 🟢 Save merged file (overwrite original path!)
merged_df.write.mode("overwrite").parquet("/user/tejashree/project/data/processed/cleaned_steam_reviews.parquet")

print("✅ Merged sentiment with cleaned reviews and saved back to HDFS.")
