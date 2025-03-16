import requests
import pandas as pd
import time
import statistics
from datasets import load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset("DigiGreen/Kenya_Agri_queries",streaming=True)
df = pd.DataFrame(dataset["train"])  # Convert to DataFrame (assuming "train" split)
df = df[['query', 'response']].dropna()

# Limit to first 1000 rows
df = df.head(100)

# OpenAI Inference API Endpoint
API_URL = "http://localhost:40000/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

# Function to send data to OpenAI Inference API
def send_request(payload):
    start_time = time.time()
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    elapsed_time = time.time() - start_time
    return response.json(), elapsed_time

# Store processing times
classifier_times = []
extractor_times = []
relevance_times = []

### FIRST LOOP: Classifier Requests
print("\nStarting Classifier Requests...")
for index, row in df.iterrows():
    query, response = row["query"], row["response"]
    classifier_payload = {
        "model": "classifier",
        "messages": [
            {"role": "system", "content": "Classify if the response fully answers the question. Output 1 if yes, 0 otherwise."},
            {"role": "user", "content": f"Q: {query}\nA: {response}"}
        ],
        "max_tokens": 5,
        "temperature": 0,
        "seed": 42
    }
    classifier_result, classifier_time = send_request(classifier_payload)
    classifier_times.append(classifier_time)
    print(f"Query {index+1}: Classifier Time = {classifier_time:.4f} sec")

print("\nClassifier Requests Completed!")

### SECOND LOOP: Extractor Requests
print("\nStarting Extractor Requests...")
for index, row in df.iterrows():
    response = row["response"]
    extractor_payload = {
        "model": "extractor",
        "messages": [
            {"role": "system", "content": "Extract agricultural facts from the response."},
            {"role": "user", "content": response}
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "seed": 42
    }
    extractor_result, extractor_time = send_request(extractor_payload)
    extractor_times.append(extractor_time)
    print(f"Query {index+1}: Extractor Time = {extractor_time:.4f} sec")

print("\nExtractor Requests Completed!")

### THIRD LOOP: Relevance Requests
print("\nStarting Relevance Requests...")
for index, row in df.iterrows():
    query, response = row["query"], row["response"]
    relevance_payload = {
        "model": "relevance",
        "messages": [
            {"role": "system", "content": "Determine if the fact is relevant to the query. Output 1 if yes, 0 otherwise."},
            {"role": "user", "content": f"Q: {query}\nA: {response}"}
        ],
        "max_tokens": 5,
        "temperature": 0,
        "seed": 42
    }
    relevance_result, relevance_time = send_request(relevance_payload)
    relevance_times.append(relevance_time)
    print(f"Query {index+1}: Relevance Time = {relevance_time:.4f} sec")

print("\nRelevance Requests Completed!")

# Print Average Processing Times
# Save the results to a text file
with open("processing_times.txt", "w") as file:
    file.write("Average Processing Times:\n")
    file.write(f"Classifier: {statistics.mean(classifier_times):.4f} sec\n")
    file.write(f"Extractor: {statistics.mean(extractor_times):.4f} sec\n")
    file.write(f"Relevance: {statistics.mean(relevance_times):.4f} sec\n")

# Print results to console
print("\nAverage Processing Times:")
print(f"Classifier: {statistics.mean(classifier_times):.4f} sec")
print(f"Extractor: {statistics.mean(extractor_times):.4f} sec")
print(f"Relevance: {statistics.mean(relevance_times):.4f} sec")

print("\nProcessing times saved to 'processing_times.txt'")

