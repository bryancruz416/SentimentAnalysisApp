import streamlit as st
import pandas as pd
import re
from bs4 import BeautifulSoup
import requests
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load 3-class sentiment model ---
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return tokenizer, model

sentiment_tokenizer, sentiment_model = load_sentiment_model()

labels_map = {
    'LABEL_0': 'Negative',
    'LABEL_1': 'Neutral',
    'LABEL_2': 'Positive'
}

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def analyze_sentiments(texts):
    results = []
    for text in texts:
        encoded_input = sentiment_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            output = sentiment_model(**encoded_input)
        scores = torch.nn.functional.softmax(output.logits, dim=1)[0]
        label_id = torch.argmax(scores).item()
        label = sentiment_model.config.id2label[label_id]
        results.append(labels_map[label])
    return results

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

def perform_kmeans(embeddings, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(embeddings)

@st.cache_resource(show_spinner=False)
def perform_topic_modeling(texts, embeddings):
    umap_model = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    topic_model = BERTopic(embedding_model=embedding_model, umap_model=umap_model)
    topics, _ = topic_model.fit_transform(texts, embeddings)
    return topics, topic_model

def process_texts(text_list, n_clusters):
    texts_clean = [clean_text(t) for t in text_list if t.strip()]
    if not texts_clean:
        return None, None
    sentiments = analyze_sentiments(texts_clean)
    embeddings = embedding_model.encode(texts_clean, show_progress_bar=False)
    clusters = perform_kmeans(embeddings, n_clusters=n_clusters)
    topics, topic_model = perform_topic_modeling(texts_clean, embeddings)
    df = pd.DataFrame({
        "Text": texts_clean,
        "Sentiment": sentiments,
        "Cluster": clusters,
        "Topic": topics
    })
    return df, topic_model

def plot_distribution(df, column, title):
    plt.figure(figsize=(8, 4))
    sns.countplot(x=column, data=df, order=df[column].value_counts().index, palette="pastel")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Count")
    st.pyplot(plt.gcf())
    plt.clf()

st.set_page_config(layout="wide")
st.title("🤖 Transformer Sentiment + Topic Modeling + KMeans Clustering")

tab1, tab2 = st.tabs(["📄 Upload/Input", "🌐 Web Scraping"])

with tab1:
    st.subheader("Upload CSV or Enter Text Manually")
    uploaded_file = st.file_uploader("Upload a CSV file with text", type="csv")
    n_clusters = st.number_input("Number of KMeans clusters", min_value=2, max_value=10, value=3, step=1)

    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
        column = st.selectbox("Select the text column", df_input.columns)
        input_texts = df_input[column].dropna().astype(str).tolist()
    else:
        text_input = st.text_area("Or paste your text (one paragraph per line):")
        input_texts = text_input.strip().split('\n') if text_input else []

    if input_texts and st.button("Run Analysis on Input Text"):
        with st.spinner("Processing..."):
            df_result, topic_model = process_texts(input_texts, n_clusters)
        if df_result is None or topic_model is None:
            st.warning("No valid text to process.")
        else:
            st.subheader("📊 Analysis Results")
            st.dataframe(df_result)

            st.subheader("📈 Sentiment Distribution")
            plot_distribution(df_result, "Sentiment", "Sentiment Distribution")

            st.subheader("📈 KMeans Clusters Distribution")
            plot_distribution(df_result, "Cluster", "KMeans Cluster Distribution")

            st.subheader("📈 Topic Frequency")
            plot_distribution(df_result, "Topic", "Topic Frequency")

            st.subheader("📌 Topics and Top Words")
            unique_topics = sorted(df_result["Topic"].unique())
            for topic_num in unique_topics:
                st.markdown(f"**Topic {topic_num}**:")
                topic_info = topic_model.get_topic(topic_num)
                if topic_info:
                    st.write(", ".join([word for word, _ in topic_info]))
                else:
                    st.write("No topic words found.")

            st.subheader("🧠 Topic Visualization")
            fig = topic_model.visualize_topics()
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Web Scraping + Sentiment & Topic Modeling")
    url = st.text_input("Enter a webpage URL to scrape text from:")
    n_clusters_scrape = st.number_input("Number of KMeans clusters (web scraping)", min_value=2, max_value=10, value=3, step=1, key='web_clusters')

    if url and st.button("Scrape & Analyze"):
        with st.spinner(f"Scraping {url} ..."):
            try:
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = soup.find_all('p')
                text_blocks = [p.get_text() for p in paragraphs if len(p.get_text()) > 30]

                if not text_blocks:
                    st.warning("No paragraphs with enough text found on the page.")
                else:
                    df_result, topic_model = process_texts(text_blocks, n_clusters_scrape)
                    if df_result is None or topic_model is None:
                        st.warning("No valid text to process.")
                    else:
                        st.subheader("📊 Scraped Text Analysis")
                        st.dataframe(df_result)

                        st.subheader("📈 Sentiment Distribution")
                        plot_distribution(df_result, "Sentiment", "Sentiment Distribution")

                        st.subheader("📈 KMeans Clusters Distribution")
                        plot_distribution(df_result, "Cluster", "KMeans Cluster Distribution")

                        st.subheader("📈 Topic Frequency")
                        plot_distribution(df_result, "Topic", "Topic Frequency")

                        st.subheader("📌 Topics and Top Words")
                        unique_topics = sorted(df_result["Topic"].unique())
                        for topic_num in unique_topics:
                            st.markdown(f"**Topic {topic_num}**:")
                            topic_info = topic_model.get_topic(topic_num)
                            if topic_info:
                                st.write(", ".join([word for word, _ in topic_info]))
                            else:
                                st.write("No topic words found.")

                        st.subheader("🧠 Topic Visualization")
                        fig = topic_model.visualize_topics()
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Failed to scrape or process: {e}")
