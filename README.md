# 🎮 GameSphere Sentiment Analysis Integration  
**Guide Purpose:** Integrate sentiment analysis into game recommendations.  
**Key Outcome:** ✅ Display sentiment scores + review insights in Streamlit app.  

## 🚩 Implementation Steps  
**1️⃣ Download Dataset:**  
[steam_sentiment_final_batched.parquet](https://drive.google.com/file/d/1n4JgzbWpr06Pwrxty-1_NJdRXzmqULxc/view?usp=sharing)  
**Columns:**  
| app_id (Game ID) | app_name (Name) | review_id (Unique ID) | cleaned_review (Text) | sentiment_label (POSITIVE/NEGATIVE) | sentiment_score (0-1 Confidence) |  

**2️⃣ HDFS Upload:**  
```bash
`hdfs dfs -mkdir -p /user/tejashree/project/data/processed && hdfs dfs -put steam_sentiment_final_batched.parquet /user/tejashree/project/data/processed/cleaned_steam_reviews_with_sentiments.parquet`
```
**Path:** `/user/tejashree/project/data/processed/cleaned_steam_reviews_with_sentiments.parquet`

**3️⃣ Merge Data:**  
`spark-submit merge_sentiment.py`  
**Merged Output:** `/user/tejashree/project/data/processed/cleaned_steam_reviews_merged.parquet`  

**4️⃣ Configure App3.py:**  
Replace with: `df = spark.read.parquet("/user/tejashree/project/data/processed/cleaned_steam_reviews_merged.parquet")`  

**5️⃣ Launch App:**  
`streamlit run app3.py` → Access at `http://localhost:8501`  

**📊 Live Output Includes:**  
- User profile summary (reviews/games/playtime)  
- Game recommendations  
- Per-game: Avg sentiment score + % positive/negative reviews  

**🎉 Example Output:**  
| Game | Rating | Sentiment | Positive% | Negative% |  
|------|--------|-----------|-----------|-----------|  
| Hades| 6.80   | 0.98      | 93%       | 7%        |  
| Hollow Knight | 6.50 | 0.95 | 90% | 10% |  
