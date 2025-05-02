from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, avg

# Initialize Spark
spark = SparkSession.builder.appName("ComputeSentimentSummary").getOrCreate()

# Load sentiment parquet
df = spark.read.parquet("/user/tejashree/project/data/processed/steam_sentiment_final_batched.parquet") 

# Add numeric sentiment for easier aggregation
df = df.withColumn("is_positive", when(col("sentiment_label") == "POSITIVE", 1).otherwise(0))
df = df.withColumn("is_negative", when(col("sentiment_label") == "NEGATIVE", 1).otherwise(0))

# Group by app_id and app_name
summary = df.groupBy("app_id", "app_name").agg(
    avg("sentiment_score").alias("avg_sentiment_score"),
    count("*").alias("total_reviews"),
    count(when(col("is_positive") == 1, True)).alias("positive_reviews"),
    count(when(col("is_negative") == 1, True)).alias("negative_reviews")
)

# Calculate percentages
summary = summary.withColumn("percent_positive", (col("positive_reviews") / col("total_reviews")) * 100)
summary = summary.withColumn("percent_negative", (col("negative_reviews") / col("total_reviews")) * 100)

# Save summary table
summary.write.mode("overwrite").parquet("/user/tejashree/project/data/processed/steam_sentiment_summary.parquet")

print("✅ Sentiment summary saved to /user/tejashree/project/data/processed/steam_sentiment_summary.parquet")
