# 🎮 GameSphere Sentiment Analysis Integration

This guide explains how to integrate the **sentiment analysis results** into the GameSphere recommendation system.

✅ By following these steps, you'll be able to display sentiment scores and review insights alongside game recommendations in the Streamlit app.

---

## 🚩 **Steps to Run the App with Sentiment Analysis**

### 1️⃣ **Download sentiment dataset**

➡️ Download the processed sentiment dataset from Google Drive:  
👉 [**https://drive.google.com/file/d/1n4JgzbWpr06Pwrxty-1_NJdRXzmqULxc/view?usp=sharing**]

The file is named:steam_sentiment_final_batched.parquet


This file contains:

| Column            | Description                              |
|------------------|------------------------------------------|
| app_id            | Game App ID                              |
| app_name          | Game Name                                |
| review_id         | Unique Review ID                         |
| cleaned_review    | Cleaned review text                      |
| sentiment_label   | Predicted sentiment label (POSITIVE/NEGATIVE) |
| sentiment_score   | Confidence score for sentiment label     |

---

### 2️⃣ **Upload file to HDFS**

Upload the downloaded file to HDFS into the project directory:

```bash
hdfs dfs -mkdir -p /user/tejashree/project/data/processed
hdfs dfs -put steam_sentiment_final_batched.parquet /user/tejashree/project/data/processed/cleaned_steam_reviews_with_sentiments.parquet


✅ This saves the file as: /user/tejashree/project/data/processed/cleaned_steam_reviews_with_sentiments.parquet

3️⃣ Run the merge_sentiment.py script
This script merges the sentiment dataset into the cleaned reviews using review_id.

Run the script via Spark: spark-submit merge_sentiment.py


✅ Output will be saved as: /user/tejashree/project/data/processed/cleaned_steam_reviews_merged.parquet
This merged file includes the sentiment columns.

4️⃣ Verify app3.py is using merged dataset
Open app3.py and ensure the following line points to the merged dataset:

df = spark.read.parquet("/user/tejashree/project/data/processed/cleaned_steam_reviews_merged.parquet")
✅ This ensures the app reads the merged dataset with sentiment info.

5️⃣ Run the Streamlit app
Launch the app:

streamlit run app3.py
Open http://localhost:8501 in your browser.

✅ App will display:

User profile summary (total reviews, games, playtime)

Game recommendations

Sentiment analysis summary per game
(average sentiment score, % positive reviews)

🎉 Expected App View
When user enters their Steam ID:

🖼️ Example:

|
|

App will show:

Recommended Game	Predicted Rating	Avg Sentiment Score	% Positive Reviews Negative Reviews
Hades	6.80	0.98	93%
Hollow Knight	6.50	0.95	90%

✅ Recommendation list now enriched with sentiment insights.


