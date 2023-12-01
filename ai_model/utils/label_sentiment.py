from transformers import pipeline
import pandas as pd

# Load the dataset
df = pd.read_excel("./data/raw.xlsx")
titles = df["title"].tolist()

# Load the sentiment analysis model
sentiment_model = pipeline(
    "sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment"
)


# Function to get sentiment label
def get_label(title):
    try:
        label = sentiment_model(title)[0]["label"]
        # Adjust these conditions based on the actual labels output by the model
        if "POSITIVE" in label:
            return 1
        elif "NEGATIVE" in label:
            return -1
        else:
            return 0
    except Exception as e:
        print(f"Error processing title: {title}, Error: {e}")
        return None


# Process in batches
batch_size = 100  # Adjust batch size based on your memory capacity
for i in range(0, len(titles), batch_size):
    batch_titles = titles[i : i + batch_size]
    df.loc[i : i + batch_size - 1, "sentiment"] = [
        get_label(title) for title in batch_titles
    ]
    # Periodically save the progress
    df.to_csv("./data/progress_labeled_data.csv", index=False)

# Save the final output
df.to_csv("./data/final_labeled_data.csv", index=False)
