from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd

# Load FinancialPhraseBank dataset
dataset = load_dataset("financial_phrasebank", "sentences_50agree", trust_remote_code=True)

data = dataset["train"]

# Map label IDs to strings
label_map = {0: "negative", 1: "neutral", 2: "positive"}
df = pd.DataFrame(data)
df["label_text"] = df["label"].map(label_map)

# Print sample
print(df.sample(5))

# Load FinBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")

# Tokenize function
def tokenize_function(example):
    return tokenizer(
        example["sentence"],
        padding="max_length",
        truncation=True,
        max_length=64  # consistent size for all
    )

# Apply tokenization
tokenized_dataset = data.map(tokenize_function, batched=True)

# Remove raw text, format for PyTorch
tokenized_dataset = tokenized_dataset.remove_columns(["sentence"])
tokenized_dataset.set_format("torch")

# Train/test split and save
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)
tokenized_dataset.save_to_disk("processed_finphrase_dataset")

print("✅ Tokenization complete and dataset saved.")
