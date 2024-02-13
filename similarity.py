# -------------------------------------------------------------------------
# AUTHOR: Ardavan Sherafat
# FILENAME: similarity
# SPECIFICATION: Program to calcualte cosine similarity and print the highest
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: 30 minutes
# -----------------------------------------------------------*/

# Importing some Python libraries
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# Defining the documents
doc1 = "soccer is my favorite sport"
doc2 = "I like sports and my favorite one is soccer"
doc3 = "support soccer at the olympic games"
doc4 = "I do like soccer, my favorite sport in the olympic games"

# Use the following words as terms to create your document-term matrix
# [soccer, favorite, sport, like, one, support, olympic, games]
# --> Add your Python code here

docs = [doc1, doc2, doc3, doc4]
dict = {'soccer': 0, 'favorite': 1, 'sport': 2, 'like': 3, 'one': 4, 'support': 5, 'olympic': 6, 'games': 7}
dataset = np.zeros((len(docs), len(dict)))

for i, doc in enumerate(docs):
    dset = np.zeros(len(dict))
    for word in doc.split():
        cleaned_word = re.sub('[^a-zA-Z]', '', word).lower()
        if cleaned_word in dict:
            dset[dict[word]] += 1
    dataset[i, :]= dset    

# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors only
# Use cosine_similarity([X, Y, Z]) to calculate the pairwise similarities between multiple vectors
# --> Add your Python code here

# 2 vectors only
max_similarity = float('-inf')
indices = [0,1]
for x in range(len(dataset)):
    for y in range(x + 1, len(dataset)):
        similarity_score = cosine_similarity([dataset[x]], [dataset[y]])
        if similarity_score > max_similarity:
            max_similarity = similarity_score
            indices = [x, y]

# multiple vectors
similarity_matrix = cosine_similarity(dataset)
np.fill_diagonal(similarity_matrix, 0)
max_similarity = np.max(similarity_matrix)
indices = np.where(similarity_matrix == max_similarity)[0]

# Print the highest cosine similarity following the information below
# The most similar documents are: doc1 and doc2 with cosine similarity = x
# --> Add your Python code here
print(f'The most similar documents are: doc{indices[0] + 1} and doc{indices[1] + 1} with cosine similarity = {max_similarity.squeeze()}')
