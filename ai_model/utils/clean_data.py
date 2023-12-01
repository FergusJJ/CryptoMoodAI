import json
import csv

# Define the JSON file path
json_file_path = "./data/labelled.json"
csv_file_path = "./data/labelled.csv"

# Read the JSON data
with open(json_file_path, "r") as file:
    data = json.load(file)

# Prepare to write to CSV
with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "text", "sentiment"])  # Write the header

    for i, entry in enumerate(data):
        text_key = "1b871c94-6e9c-4342-abe5-0908b0432c78"
        sentiment_key = "d63b79f8-f5da-47d9-ae53-26373a9d7b3d"

        # Extract relevant information
        text = entry["questions"].get(text_key)
        sentiment = entry["questions"].get(sentiment_key)

        # Write to CSV only if both text and sentiment are available
        if text and sentiment:
            writer.writerow([i, text, sentiment])

print("CSV file has been created.")
