import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from bertopic import BERTopic
from sklearn.cluster import KMeans
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

# Load models globally to avoid reloading every run
@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    return sentiment_model, sentence_model

sentiment_model, sentence_model = load_models()

def classify_sentiment(texts):
    # Model returns labels 'LABEL_0' etc, map to Negative/Neutral/Positive
    labels_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
    results = sentiment_model(texts, truncation=True)
    sentiments = [labels_map[r['label']] for r in results]
    return sentiments

def embed_texts(texts):
    return sentence_model.encode(texts, show_progress_bar=False)

def topic_and_cluster_analysis(texts, n_clusters=3):
    # Topic modeling with BERTopic
    topic_model = BERTopic(verbose=False)
    topics, probs = topic_model.fit_transform(texts)

    # Embed texts for clustering
    embeddings = embed_texts(texts)

    # Dimensionality reduction for visualization
    reducer = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    return topics, topic_model, clusters, reduced_embeddings

def plot_sentiment_distribution(sentiments):
    df = pd.DataFrame(sentiments, columns=['sentiment'])
    fig = px.histogram(df, x='sentiment', color='sentiment',
                       category_orders={"sentiment": ["negative", "neutral", "positive"]},
                       title="Sentiment Distribution")
    st.plotly_chart(fig, use_container_width=True)

def plot_clusters_2d(reduced_embeddings, clusters, sentiments):
    df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
    df['cluster'] = clusters.astype(str)
    df['sentiment'] = sentiments
    fig = px.scatter(df, x='x', y='y', color='cluster', symbol='sentiment',
                     title="K-Means Clusters with Sentiment as Symbol",
                     labels={"cluster": "Cluster", "sentiment": "Sentiment"})
    st.plotly_chart(fig, use_container_width=True)

def plot_topics(topic_model):
    fig = topic_model.visualize_topics()
    st.plotly_chart(fig, use_container_width=True)

def scrape_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Get all paragraph texts
        paragraphs = soup.find_all('p')
        text = "\n".join(p.get_text() for p in paragraphs)
        return text
    except Exception as e:
        st.error(f"Error scraping the URL: {e}")
        return ""

def main():
    st.set_page_config(page_title="Sentiment & Topic Analysis", layout="wide")
    st.title("Sentiment Analysis with Topic Modeling & K-Means Clustering")

    tab1, tab2 = st.tabs(["Upload or Paste Text", "Web Scraping + Analysis"])

    with tab1:
        st.header("Input Text Data")
        uploaded_file = st.file_uploader("Upload CSV with a text column", type=["csv"])
        text_column = None
        texts = []

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of data:")
            st.dataframe(df.head())
            text_column = st.selectbox("Select text column", options=df.columns)
            if text_column:
                texts = df[text_column].dropna().astype(str).tolist()

        if not texts:
            text_input = st.text_area("Or paste text (each paragraph/newline treated as a separate text)", height=200)
            if text_input.strip():
                texts = [line.strip() for line in text_input.split('\n') if line.strip()]

        if texts:
            n_clusters = st.slider("Number of K-Means clusters", min_value=2, max_value=10, value=3)
            if st.button("Analyze Text"):
                with st.spinner("Analyzing..."):
                    sentiments = classify_sentiment(texts)
                    topics, topic_model, clusters, reduced_embeddings = topic_and_cluster_analysis(texts, n_clusters)

                st.subheader("Sentiment Distribution")
                plot_sentiment_distribution(sentiments)

                st.subheader("Topic Modeling Visualization")
                plot_topics(topic_model)

                st.subheader("K-Means Clustering Visualization")
                plot_clusters_2d(reduced_embeddings, clusters, sentiments)

    with tab2:
        st.header("Web Scraping + Sentiment & Topic Analysis")
        url = st.text_input("Enter URL to scrape text from")
        n_clusters_ws = st.slider("Number of K-Means clusters (Web Scraping)", min_value=2, max_value=10, value=3)

        if st.button("Scrape and Analyze"):
            if url.strip() == "":
                st.warning("Please enter a URL first")
            else:
                with st.spinner("Scraping and analyzing..."):
                    scraped_text = scrape_text_from_url(url)
                    if scraped_text:
                        texts_ws = [line.strip() for line in scraped_text.split('\n') if line.strip()]
                        sentiments_ws = classify_sentiment(texts_ws)
                        topics_ws, topic_model_ws, clusters_ws, reduced_embeddings_ws = topic_and_cluster_analysis(texts_ws, n_clusters_ws)

                        st.subheader("Scraped Text Preview")
                        st.write(scraped_text[:1000] + ("..." if len(scraped_text) > 1000 else ""))

                        st.subheader("Sentiment Distribution (Scraped Text)")
                        plot_sentiment_distribution(sentiments_ws)

                        st.subheader("Topic Modeling Visualization (Scraped Text)")
                        plot_topics(topic_model_ws)

                        st.subheader("K-Means Clustering Visualization (Scraped Text)")
                        plot_clusters_2d(reduced_embeddings_ws, clusters_ws, sentiments_ws)

if __name__ == "__main__":
    main()
import os
import openai
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
@st.cache_data(show_spinner="Generating summary...", ttl=3600)
def summarize_cluster(texts, cluster_num):
    prompt = f"""
You are a consumer insights analyst. Summarize the common themes, tone, and user sentiment from these texts in Cluster {cluster_num}. 
Include:
- A 2-3 sentence summary.
- 3 to 5 bullet points of insights.

Texts:
{chr(10).join(texts[:100])}
"""
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"Error generating summary: {str(e)}"
st.subheader("Cluster Segment Summaries")

for cluster_id in sorted(df_result['Cluster'].unique()):
    cluster_texts = df_result[df_result['Cluster'] == cluster_id]['Text'].tolist()
    summary = summarize_cluster(cluster_texts, cluster_id)

    with st.expander(f"🧠 Segment {cluster_id} Insights"):
        st.markdown(summary)
