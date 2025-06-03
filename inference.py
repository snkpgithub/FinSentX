from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load your trained model and tokenizer
model_path = "finbert_sentiment_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Labels (adjust if changed during training)
id2label = {0: "negative", 1: "neutral", 2: "positive"}

# Function to run prediction
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        label_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][label_id].item()
        return id2label[label_id], confidence

# Loop for user input
print("\n🧪 FinSentX – Financial Sentiment Inference")
print("Type a financial headline or type 'exit' to quit.\n")

while True:
    text = input("📥 Enter headline: ").strip()
    if text.lower() in ["exit", "quit"]:
        break
    sentiment, confidence = predict_sentiment(text)
    print(f"🔍 Sentiment: {sentiment.upper()} (Confidence: {confidence:.2f})\n")
