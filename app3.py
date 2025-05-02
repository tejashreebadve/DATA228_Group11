import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col, from_unixtime
import plotly.express as px

# 1. Start Spark session
spark = SparkSession.builder \
    .appName("Game Recommendation App with EDA") \
    .getOrCreate()

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["🎮 Game Recommendations","📊 Interactive Dashboard", "📊 EDA Insights"])

# Recommendations (output)
recommendations_df = spark.read.parquet("/user/tejashree/project/outputs/app_recommendations.parquet")

# Game mapping data
games_df = spark.read.parquet("/user/tejashree/project/data/mappings/games_mapping.parquet")

# Author mapping data
author_mapping = spark.read.parquet("/user/tejashree/project/data/mappings/author_mapping.parquet")

# Cleaned review data
df = spark.read.parquet("/user/tejashree/project/data/processed/cleaned_steam_reviews.parquet")

# Explode recommendations
exploded_recs = recommendations_df.withColumn("rec", explode("recommendations")) \
    .select(
        col("author_index"),
        col("rec.app_index").alias("app_index"),
        col("rec.rating").alias("predicted_rating")
    )

# Join to get app names
full_recs = exploded_recs.join(games_df, on="app_index", how="inner")

# ------------------------ 🎮 Game Recommendation Section ------------------------
if option == "🎮 Game Recommendations":
    st.title("🎮 GameSphere - Game Recommendations")

    steam_id = st.text_input("Enter your Steam Author ID (author_steamid):")

    if st.button("Get Recommendations"):
        if steam_id.isdigit():
            steam_id_long = int(steam_id)
            match = author_mapping.filter(col("author_steamid") == steam_id_long).select("author_index").collect()

            if match:
                user_index = match[0]["author_index"]

                # 🧍 User Profile Summary
                st.markdown("### 🧍 User Profile Summary")
                user_stats = df.filter(col("author_steamid") == steam_id_long)
                total_reviews = user_stats.count()
                avg_playtime = user_stats.agg({"author_playtime_forever": "sum"}).collect()[0][0] / 3600
                total_games = user_stats.select("app_id").distinct().count()

                kcol1, kcol2, kcol3 = st.columns(3)
                kcol1.metric("📝 Total Reviews", f"{total_reviews}")
                kcol2.metric("🎮 Games Reviewed", f"{total_games}")
                kcol3.metric("⏱️ Total Playtime", f"{avg_playtime:.1f} hrs")

                # 🎮 Game Recommendations
                user_recs = full_recs.filter(full_recs.author_index == user_index).toPandas()

                if not user_recs.empty:
                    # ✅ Pull sentiment info per app
                    sentiment_pd = df.select("app_name", "sentiment_label", "sentiment_score").toPandas()

                    sentiment_summary = (
                        sentiment_pd.groupby("app_name")
                        .agg(
                            avg_sentiment_score=("sentiment_score", "mean"),
                            positive_reviews=("sentiment_label", lambda x: (x == "POSITIVE").sum()),
                            negative_reviews=("sentiment_label", lambda x: (x == "NEGATIVE").sum())
                        )
                        .reset_index()
                    )

                    # ✅ Merge with recommendations
                    user_recs = user_recs.merge(sentiment_summary, on="app_name", how="left")

                    st.success(f"Top {len(user_recs)} recommended games for Steam ID {steam_id}:")
                    
                    # ✅ Display text insights for each recommendation
                    for index, row in user_recs.iterrows():
                        st.markdown(f"""
                        ### 🎮 **{row['app_name']}**
                        - **Predicted Rating:** {row['predicted_rating']:.2f}
                        - **Avg Sentiment Score:** {row['avg_sentiment_score']:.2f}
                        - ✅ **{int(row['positive_reviews'])} positive reviews**
                        - ❌ **{int(row['negative_reviews'])} negative reviews**
                        """)

                    # ✅ Display table
                    st.dataframe(
                        user_recs[["app_name", "predicted_rating", "avg_sentiment_score", "positive_reviews", "negative_reviews"]]
                        .sort_values(by="predicted_rating", ascending=False)
                    )
                else:
                    st.warning("No recommendations found for this user.")
            else:
                st.error("Steam ID not found in the system.")
        else:
            st.warning("Please enter a valid numeric Steam ID.")

# New Interactive Dashboard Tab
if option == "📊 Interactive Dashboard":
    st.title("📊 Deep-Dive Game Insights Dashboard")

    # Extract month and app_name fields (timestamp already in TIMESTAMP format)
    df_game = df.select("app_name", "recommended", "author_playtime_forever", "timestamp_created") \
               .withColumn("month", col("timestamp_created").cast("date")) \
               .withColumn("month", col("month").substr(1, 7))

    # Get top 50 games by review count
    top_games_df = df_game.groupBy("app_name").count().orderBy(col("count").desc()).limit(50)
    top_games = [row["app_name"] for row in top_games_df.collect()]

    # Create UI filters
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_game = st.selectbox("🎮 Select a Game", top_games)
    with col2:
        available_months = df_game.select("month").distinct().orderBy(col("month").desc()).toPandas()["month"].tolist()
        selected_month = st.selectbox("📅 Select Month", available_months)

    # Filter data based on selection
    filtered_df = df_game.filter((col("app_name") == selected_game) & (col("month") == selected_month))
    filtered_pd = filtered_df.toPandas()

    if not filtered_pd.empty:
        st.markdown(f"### 🔍 Detailed Analysis for **{selected_game}** ({selected_month})")

        # Layout columns for metrics and export
        mcol1, mcol2 = st.columns([1, 1])

        with mcol1:
            total_reviews = len(filtered_pd)
            st.metric(label="📝 Total Reviews", value=f"{total_reviews}")

        with mcol2:
            csv = filtered_pd.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"{selected_game}_{selected_month}_analysis.csv",
                mime='text/csv'
            )

        # Visuals side by side
        vcol1, vcol2 = st.columns([1, 1])


        with vcol1:
            filtered_pd["playtime_hours"] = filtered_pd["author_playtime_forever"] / 60
            fig = px.histogram(
                filtered_pd,
                x="playtime_hours",
                nbins=30,
                title="Playtime Distribution (Hours)",
                labels={"playtime_hours": "Playtime (hrs)", "count": "Number of Users"},
                color_discrete_sequence=["#00cc96"]
            )
            fig.update_layout(yaxis_title="Number of Players")
            st.plotly_chart(fig, use_container_width=True)


        with vcol2:
            rec_counts = filtered_pd["recommended"].value_counts().reset_index()
            rec_counts.columns = ["Recommendation", "Count"]
            rec_counts["Recommendation"] = rec_counts["Recommendation"].replace({True: "Recommended", False: "Not Recommended"})
            fig_pie = px.pie(
                rec_counts,
                values="Count",
                names="Recommendation",
                title="Recommendation Ratio",
                color_discrete_sequence=["#636efa", "#ef553b"]
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    else:
        st.warning("No data available for the selected game and month.")

# ------------------------ 📊 EDA Section ------------------------
elif option == "📊 EDA Insights":
    st.title("📊 Exploratory Data Analysis (EDA) - Game Reviews")

    # --- Most Reviewed Games ---
    st.subheader("📌 Top 20 Most Reviewed Games")
    app_names = df.groupBy("app_name").count()
    app_names_count = app_names.orderBy(col("count").desc()).limit(20)
    app_counts_pd = app_names_count.toPandas().sort_values(by="count", ascending=True)
    fig1 = px.bar(app_counts_pd, x="count", y="app_name", orientation='h', color="count",
                  labels={"count": "Number of Reviews", "app_name": "Game Name"},
                  title="Top 20 Most Reviewed Games")
    st.plotly_chart(fig1)

    # --- Playtime Distribution ---
    st.subheader("⏰ Distribution of Playtime (Hours)")
    playtime_df = df.select((col("author_playtime_forever") / 60).alias("playtime_hours")) \
                    .filter(col("playtime_hours").isNotNull())
    playtime_pd = playtime_df.sample(False, 0.01, seed=42).toPandas()
    fig2 = px.histogram(playtime_pd, x='playtime_hours', nbins=50,
                        title="Playtime Distribution",
                        labels={'playtime_hours': 'Playtime (Hours)'})
    st.plotly_chart(fig2)

    # --- Language Breakdown ---
    st.subheader("🌐 Review Language Breakdown")
    lang_df = df.groupBy("language").count().orderBy(col("count").desc()).limit(10)
    lang_pd = lang_df.toPandas()
    fig3 = px.pie(lang_pd, values="count", names="language",
                  title="Top 10 Review Languages")
    st.plotly_chart(fig3)

    # --- Recommendation Ratio ---
    
    st.subheader("🔥 Top 10 Highly Recommended Games")

    # Filter only recommended reviews (True) and count by app_name
    true_counts = df.filter(col("recommended") == True).groupBy("app_name").count()

    # Get top 10 recommended games
    recommended = true_counts.orderBy(col("count").desc()).limit(10)
    recommended_apps = recommended.toPandas().sort_values(by="count", ascending=True)

    # Plot as horizontal bar chart
    fig = px.bar(recommended_apps, x="count", y="app_name", orientation='h',
                labels={"count": "Recommendation Count", "app_name": "Game Name"},
                title="Top 10 Most Recommended Games",
                color="count")

    st.plotly_chart(fig)


    # --- Review Trend Over Time ---
    st.subheader("📈 Review Volume Over Time")
    df_dates = df.select((col("timestamp_created")).alias("ts"))
    df_dates = df_dates.withColumn("month", from_unixtime(col("ts"), "yyyy-MM"))
    monthly_df = df_dates.groupBy("month").count().orderBy("month")
    monthly_pd = monthly_df.toPandas()
    fig5 = px.line(monthly_pd, x="month", y="count",
                   title="Monthly Review Volume",
                   labels={"month": "Month", "count": "Number of Reviews"})
    st.plotly_chart(fig5)
    # ⏰ Playtime Analysis for Top 50 Players
    st.subheader("⏰ Top 50 Most Invested Players by Hours Played")
    data_author = df.select("author_steamid", "steam_purchase", "author_num_games_owned", "author_playtime_forever", "author_playtime_at_review")

    author_playtime = data_author.filter(col("author_steamid") >= 76560000000000000) \
        .orderBy(col("author_playtime_forever").desc()) \
        .limit(50).toPandas()

    author_playtime["author_playtime_forever"] = author_playtime["author_playtime_forever"] / 60
    fig6 = px.bar(author_playtime, 
                x="author_playtime_forever", 
                y="author_steamid", 
                orientation="h",
                labels={"author_playtime_forever": "Playtime (hrs)", "author_steamid": "Steam ID"},
                title="Top 50 Players by Total Playtime",
                color="author_num_games_owned",
                color_continuous_scale="Viridis")
    st.plotly_chart(fig6)

    # ------------------------------------
    # 📈 Correlation Between Playtime Forever and At Review
    st.subheader("📈 Total vs. At Review Playtime Correlation")
    author_review_playtime = data_author.filter(col("author_steamid") >= 76560000000000000) \
        .select("author_playtime_forever", "author_playtime_at_review") \
        .limit(5000).toPandas()

    author_review_playtime["author_playtime_forever"] = author_review_playtime["author_playtime_forever"] / 3600
    author_review_playtime["author_playtime_at_review"] = author_review_playtime["author_playtime_at_review"] / 3600

    fig7 = px.scatter(
        author_review_playtime,
        x="author_playtime_at_review",
        y="author_playtime_forever",
        labels={"author_playtime_at_review": "Playtime at Review (hrs)", "author_playtime_forever": "Total Playtime (hrs)"},
        title="Do Users Keep Playing After Reviewing?",
        trendline="ols",
        opacity=0.6,
        color_discrete_sequence=["#636EFA"]
    )
    st.plotly_chart(fig7)

