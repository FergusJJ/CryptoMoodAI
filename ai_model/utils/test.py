from transformers import pipeline

# Load the sentiment analysis model
sentiment_model = pipeline("sentiment-analysis", model="ElKulako/cryptobert")

# Sample text
sample_text = [
    "This is terrible",
    "this is okay",
    '"Can I eat it now?"',
    "this is the worst thing ever",
    "this is not so bad",
    "bitcoin",
    "ethereum",
    "bitcoin is 50k",
    "etheruem following a bullish trend",
]

# Get the model output for the sample text
for text in sample_text:
    output = sentiment_model(text)
    print(f"in: {text}\nout: {output}")
