import ollama
import numpy as np
from numpy.linalg import norm
import os
import pandas as pd

# Directory containing the text files
directory = "/Users/carlstahmer/Workspaces/teaching/workshop_llm_with_ollama/data/plosOne/"

# Ouptut directory to save outputs
out_directory = "/Users/carlstahmer/Desktop/"

# List to store the contents of each file
documents = []
manifest = []

# Iterate through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):  # Ensure we're only reading .txt files
        manifest.append(filename)
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            documents.append(file.read())

# package the documents as a dataframe with filenames as the index
documents_df = pd.DataFrame(documents, columns=['text'])
documents_df.index = manifest

# define a function for getting text embeddings
def embed(text):
    # Function body
    response = ollama.embed(model="mxbai-embed-large", input=text)
    text_embeddings = response["embeddings"][0]
    return text_embeddings

# Call the embed funciton on all rows of the dataframe and add them
# as a new column
# x =  map(embed, documents_df['text'])
documents_df['embeddings'] = list(map(embed, documents_df['text']))

# save the dataframe to disk for future use
save_path = os.path.join(out_directory, "plosone_embeddings.csv")
documents_df.to_csv(save_path, index=True)

print("Done")



