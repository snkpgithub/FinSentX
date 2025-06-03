import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
import os

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Load model and tokenizer
MODEL_PATH = "finbert_sentiment_model"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Styling
st.set_page_config(page_title="FinSentX", layout="wide")
st.markdown(
    """
    <style>
    .main {background-color: #f7f7f7;}
    .stApp {font-family: 'Segoe UI', sans-serif;}
    .sentiment-box {
        padding: 0.75em;
        border-radius: 0.5em;
        font-weight: bold;
        font-size: 1.2em;
        text-align: center;
    }
    .positive {background-color: #d4edda; color: #155724;}
    .neutral {background-color: #fff3cd; color: #856404;}
    .negative {background-color: #f8d7da; color: #721c24;}
    </style>
    """,
    unsafe_allow_html=True
)

# Predict function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        label_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][label_id].item()
        return id2label[label_id], round(confidence, 2)

# Header
st.title("💹 FinSentX")
st.subheader("Finance Sentiment Intelligence Powered by FinBERT")

tab1, tab2 = st.tabs(["🔍 Single Headline Analysis", "📂 Bulk Sentiment from CSV"])

# --- Single Input
with tab1:
    st.markdown("Type or paste a financial news headline below:")
    user_input = st.text_area("✏️ Headline Input", height=100, placeholder="e.g. Company shares soared after quarterly earnings beat expectations...")

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            sentiment, confidence = predict_sentiment(user_input)
            color_class = sentiment.lower()
            st.markdown(f'<div class="sentiment-box {color_class}">{sentiment} — {confidence*100:.1f}% confident</div>', unsafe_allow_html=True)
        else:
            st.warning("Headline can't be empty!")

# --- Bulk Upload
with tab2:
    st.markdown("Upload a CSV with a column named `headline` to get batch sentiment predictions.")
    uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "headline" not in df.columns:
            st.error("❌ The uploaded file must contain a 'headline' column.")
        else:
            results = []
            for text in df["headline"]:
                label, conf = predict_sentiment(str(text))
                results.append({"headline": text, "sentiment": label, "confidence": conf})

            result_df = pd.DataFrame(results)
            st.success(f"✅ Processed {len(result_df)} headlines successfully.")
            st.dataframe(result_df)

            # Sentiment chart
            st.subheader("📊 Sentiment Breakdown")
            chart_data = result_df["sentiment"].value_counts().reset_index()
            chart_data.columns = ["Sentiment", "Count"]
            st.bar_chart(chart_data.set_index("Sentiment"))

            # Download button
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", csv, "sentiment_results.csv", "text/csv")
