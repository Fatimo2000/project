#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/05/2020

@author: julien and antoine
"""

# This script requires the praw package.
# For more details, see:
# https://towardsdatascience.com/scraping-reddit-data-1c0af3040768

import os  # Standard library for interacting with the operating system
import dotenv  # Library for loading environment variables from a .env file
import datetime  # Library for handling date and time
from urllib import request  # Library for URL handling
import pandas as pd  # Library for data manipulation and analysis
import numpy as np  # Library for numerical operations

# =============== PARTIE 1 =============
# =============== 1.1 : REDDIT ===============
# Library for accessing the Reddit API
import praw

# =============== 1.2 : ArXiv ===============
# Library for parsing XML data
import xmltodict

# Load Environment Variables
dotenv.load_dotenv()

# Fetch Reddit API credentials from environment variables
reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
reddit_user_agent = os.getenv("REDDIT_USER_AGENT")

# Initialize Reddit instance with provided credentials
reddit = praw.Reddit(
    client_id=reddit_client_id,
    client_secret=reddit_client_secret,
    user_agent=reddit_user_agent,
)

# Notify user that data fetching from Reddit is starting
print("Fetching data from reddit ...")
subr = reddit.subreddit("Coronavirus").hot(
    limit=100
)  # Fetch hot posts from the "Coronavirus" subreddit
textes_Reddit = []  # List to store Reddit post data

# Extract relevant information from each Reddit post
for post in subr:
    texte = post.title.replace("\n", " ")  # Clean title by removing newlines
    data = {"texte": texte, "source": "reddit"}  # Create a dictionary for the post data
    textes_Reddit.append(data)  # Append the data to the list

# Notify user that Reddit data has been downloaded
print("Reddit data downloaded.")

textes_Arxiv = []  # List to store ArXiv entry data

query = "covid"  # Define search query for ArXiv
url = (
    "http://export.arxiv.org/api/query?search_query=all:"
    + query
    + "&start=0&max_results=100"
)  # Construct the ArXiv API query URL

# Notify user that data fetching from ArXiv is starting
print("Fetching data from arxiv ...")
url_read = request.urlopen(url).read()  # Open the URL and read the data

# Decode the byte stream into a string
data = url_read.decode()
dico = xmltodict.parse(data)  # Parse the XML response into a dictionary format
docs = dico["feed"]["entry"]  # Extract entries from the parsed data

# Extract relevant information from each ArXiv entry
for d in docs:
    texte = d["title"] + ". " + d["summary"]  # Combine title and summary
    texte = texte.replace("\n", " ")  # Clean text by removing newlines
    data = {"texte": texte, "source": "arxiv"}  # Create a dictionary for the entry data
    textes_Arxiv.append(data)  # Append the data to the list

# Notify user that ArXiv data has been downloaded
print("Arxiv data downloaded.")

# Combine Reddit and ArXiv data
df_data = textes_Reddit + textes_Arxiv

# Notify user that data is being saved
print("Saving Data to output.csv")
df = pd.DataFrame(df_data)  # Create a DataFrame from the combined data
df["id"] = range(1, len(df) + 1)  # Add an ID column to the DataFrame

# Save the DataFrame to a CSV file
with open("./data/output.csv", mode="w", encoding="utf-8") as file:
    df.to_csv(file, sep="\t", index=False)  # Save as tab-separated values

# Notify user that data loading is starting
print("Loading Data from output.csv")
with open("./data/output.csv", mode="r+", encoding="utf-8") as file:
    corpus = pd.read_csv(file, sep="\t")  # Read the data from the CSV file

    print("Longueur du corpus : " + str(len(corpus)))  # Print the length of the corpus

    indices_to_drop = []  # List to store indices of documents to drop
    for index, doc in corpus.iterrows():
        # Count the number of sentences and words in each document
        print("*****************************")
        num_de_phrases = len(doc["texte"].split("."))  # Count sentences
        num_de_mots = len(doc["texte"].split(" "))  # Count words
        print(f"Nombre de phrases : {num_de_phrases}")  # Print number of sentences
        print(f"Nombre de mots : {num_de_mots}")  # Print number of words

        # Mark documents with fewer than 20 words for deletion
        if num_de_mots < 20:
            indices_to_drop.append(index)

    # Notify user that documents with fewer than 20 words are being deleted
    print("Deleting less than 20 Chars docs ...")
    corpus.drop(indices_to_drop, inplace=True)  # Drop the marked documents
    corpus.to_csv(file, index=False)  # Save the updated corpus to the file
    print(
        f"{len(indices_to_drop)} Docs has been deleted."
    )  # Print the number of deleted documents

    # Calculate and print average number of sentences and words
    nb_phrases = [len(doc["texte"].split(".")) for _, doc in corpus.iterrows()]
    print(f"Moyenne du nombre de phrases : {np.mean(nb_phrases)}")
    nb_mots = [len(doc["texte"].split(" ")) for _, doc in corpus.iterrows()]
    print(f"Moyenne du nombre de mots : {np.mean(nb_mots)}")
    print(f"Nombre total de mots dans le corpus : {np.sum(nb_mots)}")

# Print the current date and time
aujourdhui = datetime.datetime.now()
print(aujourdhui)
