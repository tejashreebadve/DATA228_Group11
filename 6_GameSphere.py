import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col, date_format
import plotly.express as px

# --- Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap');

    html, body, [class*="css"], .main, .block-container, section {
        background-color: #0e1117;
        color: #f5f5f5;
        font-family: 'Orbitron', sans-serif;
    }

    /* Global heading styling (includes st.title) */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        text-shadow: 0 0 10px #7c3aed, 0 0 20px #c084fc;
        font-weight: 800;
    }

    /* App title (legacy fallback) */
    .gamesphere-title {
        font-size: 2.8rem;
        color: #ffffff;
        text-shadow: 0 0 12px #a855f7, 0 0 22px #9333ea;
        font-weight: 800;
        margin-bottom: 1.5rem;
    }


    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: bold;
        color: #c084fc;
        background-color: #2a2a40;
        border-radius: 6px;
        margin-right: 10px;
        padding: 6px 12px;
        transition: 0.3s;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(to right, #c084fc, #7c3aed);
        color: white;
        box-shadow: 0 0 8px #c084fc;
    }

    /* Buttons */
    .stButton>button {
        background-color: #c084fc;
        color: white;
        border: none;
        font-weight: bold;
        padding: 0.6em 1.4em;
        border-radius: 10px;
        box-shadow: 0 0 12px #a855f7;
        transition: all 0.3s ease-in-out;
    }

    .stButton>button:hover {
        background-color: #a855f7;
        box-shadow: 0 0 16px #9333ea;
    }

    /* Text inputs */
    input[type="text"], textarea, .stTextInput>div>div>input, .stTextInput input {
        background-color: #1e1e2f !important;
        color: #f5f5f5 !important;
        border: 1px solid #3a3a4d !important;
        border-radius: 8px;
        padding: 0.5em;
    }

    /* Text input label */
    label, .stTextInput label {
        color: #ffb3ec !important;
        font-weight: 600;
        font-size: 1rem;
    }

    /* Metrics */
    .stMetric {
        background-color: #292942;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 0 8px rgba(0,0,0,0.5);
    }

    [data-testid="stMetricLabel"] {
        color: #aaa !important;
    }

    [data-testid="stMetricValue"] {
        color: #f5f5f5 !important;
    }

    /* DataFrame Styling */
    .stDataFrame div[data-testid="stDataFrame"] {
        background-color: #1a1a28;
        color: #f5f5f5;
        border-radius: 10px;
    }

    /* Alert messages (success, warning, error) */
    .stAlert {
        background-color: #222235 !important;
        border-left: 5px solid #c084fc !important;
        color: #e0e0e0 !important;
    }

    /* Download button */
    button[kind="download"] {
        background-color: #c084fc !important;
        color: white !important;
        border-radius: 8px;
        border: none;
        padding: 0.6em 1.4em;
        box-shadow: 0 0 10px #a855f7;
    }

    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# 1. Start Spark session
spark = SparkSession.builder \
    .appName("Game Recommendation App with EDA") \
    .getOrCreate()

# Add custom dark theme styling
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: white;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# Tabs layout
tab1, tab2, tab3 = st.tabs(["🎮 Game Recommendations", "📊 Interactive Dashboard", "📊 EDA Insights"])

# Load Data
recommendations_df = spark.read.parquet("/user/tejashree/project/outputs/app_recommendations.parquet")
games_df = spark.read.parquet("/user/tejashree/project/data/mappings/games_mapping.parquet")
author_mapping = spark.read.parquet("/user/tejashree/project/data/mappings/author_mapping.parquet")
df = spark.read.parquet("/user/tejashree/project/data/processed/cleaned_steam_reviews.parquet")
sentiment_summary = spark.read.parquet("/user/tejashree/project/data/processed/steam_sentiment_summary.parquet")  # <== new summary table

# Explode recommendations
exploded_recs = recommendations_df.withColumn("rec", explode("recommendations")) \
    .select(
        col("author_index"),
        col("rec.app_index").alias("app_index"),
        col("rec.rating").alias("predicted_rating")
    )

# Join to get app names
full_recs = exploded_recs.join(games_df, on="app_index", how="inner")

# Join sentiment summary into recommendations
full_recs_with_sentiment = full_recs.join(sentiment_summary, full_recs.app_name == sentiment_summary.app_name, "left") \
    .select(
        full_recs["app_name"],
        "predicted_rating",
        "avg_sentiment_score",
        "positive_reviews",
        "negative_reviews",
        "percent_positive",
        "percent_negative"
    )

# ------------------------  Game Recommendation Section ------------------------
# with tab1:
#     st.title("🎮 GameSphere - Game Recommendations")
#     steam_id = st.text_input("Enter your Steam Author ID (author_steamid):")

#     if st.button("Get Recommendations"):
#         if steam_id.isdigit():
#             steam_id_long = int(steam_id)
#             match = author_mapping.filter(col("author_steamid") == steam_id_long).select("author_index").collect()

#             if match:
#                 user_index = match[0]["author_index"]
#                 st.markdown("### 🧍 User Profile Summary")
#                 user_stats = df.filter(col("author_steamid") == steam_id_long)
#                 total_reviews = user_stats.count()
#                 avg_playtime = user_stats.agg({"author_playtime_forever": "sum"}).collect()[0][0] / 3600
#                 total_games = user_stats.select("app_id").distinct().count()

#                 kcol1, kcol2, kcol3 = st.columns(3)
#                 kcol1.metric("📝 Total Reviews", f"{total_reviews}")
#                 kcol2.metric("🎮 Games Reviewed", f"{total_games}")
#                 kcol3.metric("⏱️ Total Playtime", f"{avg_playtime:.1f} hrs")

#                 user_recs = full_recs.filter(full_recs.author_index == user_index).toPandas()

#                 if not user_recs.empty:
#                     st.success(f"Top {len(user_recs)} recommended games for Steam ID {steam_id}:")
#                     st.dataframe(user_recs[["app_name", "predicted_rating"]].sort_values(by="predicted_rating", ascending=False))
#                 else:
#                     st.warning("No recommendations found for this user.")
#             else:
#                 st.error("Steam ID not found in the system.")
#         else:
#             st.warning("Please enter a valid numeric Steam ID.")
with tab1:
    #st.title("🎮 GameSphere - Game Recommendations")
    st.markdown("""
    <h1 style='
        font-size: 2.8rem;
        color: #ffffff;
        text-shadow: 0 0 12px #a855f7, 0 0 22px #9333ea;
        font-weight: 800;
        margin-bottom: 1.5rem;
    '>
    🎮 GameSphere - Game Recommendations
    </h1>
    """, unsafe_allow_html=True)
  
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

                # 🎮 Game Recommendations with sentiment stats
                user_recs = full_recs_with_sentiment.filter(full_recs.author_index == user_index).toPandas()

                if not user_recs.empty:
                    user_recs_display = user_recs.sort_values(by="predicted_rating", ascending=False)
                    st.success(f"Top {len(user_recs_display)} recommended games for Steam ID {steam_id}:")
                    st.dataframe(user_recs_display[[
                        "app_name",
                        "predicted_rating",
                        "avg_sentiment_score",
                        "percent_positive",
                        "percent_negative"
                    ]])
                else:
                    st.warning("No recommendations found for this user.")
            else:
                st.error("Steam ID not found in the system.")
        else:
            st.warning("Please enter a valid numeric Steam ID.")

# ------------------------ 📊 Interactive Dashboard Section ------------------------
with tab2:
    st.title("📊 Deep-Dive Game Insights Dashboard")
    

    df_game = df.select("app_name", "recommended", "author_playtime_forever", "timestamp_created") \
               .withColumn("month", col("timestamp_created").cast("date")) \
               .withColumn("month", col("month").substr(1, 7))

    top_games_df = df_game.groupBy("app_name").count().orderBy(col("count").desc()).limit(50)
    top_games = [row["app_name"] for row in top_games_df.collect()]

    col1, col2 = st.columns([1, 1])
    with col1:
        selected_game = st.selectbox("🎮 Select a Game", top_games)
    with col2:
        available_months = df_game.select("month").distinct().orderBy(col("month").desc()).toPandas()["month"].tolist()
        selected_month = st.selectbox("📅 Select Month", available_months)

    filtered_df = df_game.filter((col("app_name") == selected_game) & (col("month") == selected_month))
    filtered_pd = filtered_df.toPandas()

    if not filtered_pd.empty:
        st.markdown(f"### 🔍 Detailed Analysis for **{selected_game}** ({selected_month})")

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

        vcol1, vcol2 = st.columns([1, 1])
        with vcol1:
            filtered_pd["playtime_hours"] = filtered_pd["author_playtime_forever"] / 60
            fig = px.histogram(filtered_pd, x="playtime_hours", nbins=30,
                               title="Playtime Distribution (Hours)",
                               labels={"playtime_hours": "Playtime (hrs)", "count": "Number of Users"},
                               color_discrete_sequence=["#00cc96"])
            fig.update_layout(yaxis_title="Number of Players")
            st.plotly_chart(fig, use_container_width=True)

        with vcol2:
            rec_counts = filtered_pd["recommended"].value_counts().reset_index()
            rec_counts.columns = ["Recommendation", "Count"]
            rec_counts["Recommendation"] = rec_counts["Recommendation"].replace({True: "Recommended", False: "Not Recommended"})
            fig_pie = px.pie(rec_counts, values="Count", names="Recommendation",
                             title="Recommendation Ratio",
                             color_discrete_sequence=["#636efa", "#ef553b"])
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("No data available for the selected game and month.")
   #---- tab3
with tab3:
    st.title("📊 Exploratory Data Analysis (EDA)")

    st.subheader("📌 Top 15 Most Reviewed Games")
    most_reviewed_pd = df.groupBy("app_name").count().orderBy(col("count").desc()).limit(15).toPandas().sort_values(by="count", ascending=True)
    fig1 = px.bar(most_reviewed_pd, x="count", y="app_name", orientation='h',
                  labels={"count": "Number of Reviews", "app_name": "Game Name"},
                  color="count", height=500)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    st.subheader("🔥 Top 15 Most Recommended Games")
    recommended_pd = df.filter(col("recommended") == True).groupBy("app_name").count() \
                       .orderBy(col("count").desc()).limit(15).toPandas().sort_values(by="count", ascending=True)
    fig3 = px.bar(recommended_pd, x="count", y="app_name", orientation='h',
                  labels={"count": "Recommendation Count", "app_name": "Game Name"},
                  color="count", height=400)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    with st.expander("🌐 Review Language Breakdown"):
        lang_pd = df.groupBy("language").count().orderBy(col("count").desc()).limit(10).toPandas()
        fig4 = px.pie(lang_pd, values="count", names="language", title="Top 10 Review Languages")
        st.plotly_chart(fig4, use_container_width=True)

    with st.expander("📈 Review Volume Over Time"):
        df_dates = df.select((col("timestamp_created")).alias("ts")).withColumn("month", date_format(col("ts"), "yyyy-MM"))
        monthly_pd = df_dates.groupBy("month").count().orderBy("month").toPandas()
        fig5 = px.line(monthly_pd, x="month", y="count", title="Monthly Review Volume")
        st.plotly_chart(fig5, use_container_width=True)


