import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud

# Set up Streamlit page config
st.set_page_config(page_title="Social Media Sentiment Dashboard", layout="wide")
st.title("📊 Social Media Sentiment & Market Trend Analysis")

# Load dataset
df = pd.read_csv("C:/Users/jayde/Downloads/powerbi.csv")

# Clean whitespace in string columns
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Sidebar filters
st.sidebar.header("🔍 Filters")
platforms = st.sidebar.multiselect("Select Platforms:", options=df["Platform"].unique(), default=df["Platform"].unique())
countries = st.sidebar.multiselect("Select Countries:", options=df["Country"].unique(), default=df["Country"].unique())

# Apply filters
filtered_df = df[(df["Platform"].isin(platforms)) & (df["Country"].isin(countries))]

# Section 1: Sentiment Distribution
st.subheader("1. Sentiment Distribution")
sentiment_counts = filtered_df["Predicted_Sentiment"].value_counts()
fig1 = px.bar(x=sentiment_counts.index, y=sentiment_counts.values,
              labels={"x": "Sentiment", "y": "Count"},
              color=sentiment_counts.index,
              title="Sentiment Distribution")
st.plotly_chart(fig1, use_container_width=True)

# Section 2: Temporal Analysis
st.subheader("2. Temporal Sentiment Trends")
temporal = filtered_df.groupby(["Year", "Month", "Predicted_Sentiment"]).size().reset_index(name="Count")
temporal["Date"] = pd.to_datetime(temporal[["Year", "Month"]].assign(DAY=1))
fig2 = px.line(temporal, x="Date", y="Count", color="Predicted_Sentiment", title="Monthly Sentiment Trend")
st.plotly_chart(fig2, use_container_width=True)

# Section 3: User Behavior Insights
st.subheader("3. User Engagement (Likes & Retweets)")
engagement_df = filtered_df.groupby("Predicted_Sentiment")[["Likes", "Retweets"]].mean().reset_index()
fig3 = px.bar(engagement_df, x="Predicted_Sentiment", y=["Likes", "Retweets"],
              barmode="group", title="Average Likes & Retweets by Sentiment")
st.plotly_chart(fig3, use_container_width=True)

# Section 4: Platform-Specific Sentiment
st.subheader("4. Platform-Based Sentiment")
platform_sentiment = filtered_df.groupby(["Platform", "Predicted_Sentiment"]).size().reset_index(name="Count")
fig4 = px.bar(platform_sentiment, x="Platform", y="Count", color="Predicted_Sentiment", barmode="group",
              title="Sentiment Distribution Across Platforms")
st.plotly_chart(fig4, use_container_width=True)

# Section 5: Hashtag Trends
st.subheader("5. Trending Hashtags")
hashtag_series = filtered_df["Hashtags"].str.split().explode()
top_hashtags = hashtag_series.value_counts().head(20).reset_index()
top_hashtags.columns = ["Hashtag", "Count"]
fig5 = px.bar(top_hashtags, x="Hashtag", y="Count", title="Top 20 Trending Hashtags")
st.plotly_chart(fig5, use_container_width=True)

# Section 5.1: Word Cloud
st.subheader("5.1 Word Cloud of Frequently Used Words")
all_text = " ".join(filtered_df["Text"].dropna().astype(str).values)
wordcloud = WordCloud(width=1000, height=400, background_color='white').generate(all_text)
fig_wc, ax = plt.subplots(figsize=(15, 6))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig_wc)

# Section 6: Geographical Sentiment Distribution
st.subheader("6. Sentiment by Country")
geo_sentiment = filtered_df.groupby(["Country", "Predicted_Sentiment"]).size().reset_index(name="Count")
fig6 = px.choropleth(geo_sentiment, locations="Country", locationmode="country names",
                     color="Count", hover_name="Country",
                     animation_frame="Predicted_Sentiment",
                     title="Sentiment Distribution by Country",
                     width=14000, height=800,
                     color_continuous_scale=px.colors.sequential.YlOrRd)
st.plotly_chart(fig6, use_container_width=False)

# Section 7: Influential Users
st.subheader("7. Influential Users")
user_post_count = filtered_df["User"].value_counts().head(10).reset_index()
user_post_count.columns = ["User", "Posts"]
fig7 = px.bar(user_post_count, x="User", y="Posts", title="Top 10 Active Users")
st.plotly_chart(fig7, use_container_width=True)

# Section 8: Cross Analysis (Sentiment over Time by Platform)
st.subheader("8. Cross Analysis: Time-Series Sentiment by Platform")
cross = filtered_df.groupby(["Year", "Month", "Platform", "Predicted_Sentiment"]).size().reset_index(name="Count")
cross["Date"] = pd.to_datetime(cross[["Year", "Month"]].assign(DAY=1))
fig8 = px.line(cross, x="Date", y="Count", color="Platform", line_dash="Predicted_Sentiment",
               title="Sentiment Trends by Platform Over Time")
st.plotly_chart(fig8, use_container_width=True)

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Data Source: powerbi.csv")
