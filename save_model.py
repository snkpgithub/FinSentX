from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model from your best checkpoint
model = AutoModelForSequenceClassification.from_pretrained("finbert_output/checkpoint-486")

# Load tokenizer from the original model
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")

# Save both to a clean, final folder
model.save_pretrained("finbert_sentiment_model")
tokenizer.save_pretrained("finbert_sentiment_model")

print("✅ Model and tokenizer saved to 'finbert_sentiment_model/'")
