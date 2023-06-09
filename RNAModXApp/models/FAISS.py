import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Load pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Read the text file
with open("state_of_the_union.txt", "r", encoding="utf-8") as file:
    texts = file.readlines()

# Tokenize and encode the texts
encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Get the input IDs
input_ids = encoded_texts["input_ids"]

# Generate the embeddings
with torch.no_grad():
    outputs = model(input_ids)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # Use the CLS token embeddings

# Create an index
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance is used

# Add the embeddings to the index
index.add(embeddings)

# Generate query embeddings
query_text = "This is a query text."
encoded_query = tokenizer([query_text], padding=True, truncation=True, return_tensors="pt")
query_input_ids = encoded_query["input_ids"]

with torch.no_grad():
    query_output = model(query_input_ids)
    query_embedding = query_output.last_hidden_state[:, 0, :].numpy()

# Perform similarity search
k = 5  # Number of nearest neighbors to retrieve
distances, indices = index.search(query_embedding, k)

# Print the nearest neighbors
print("Nearest neighbors:")
for i in range(k):
    index = indices[0][i]
    distance = distances[0][i]
    neighbor_text = texts[index]
    print(f"Index: {index}, Distance: {distance}, Text: {neighbor_text}")
