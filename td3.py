#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/05/2020

@author: julien and antoine
"""

# needs praw package, cf.:
# https://towardsdatascience.com/scraping-reddit-data-1c0af3040768
import os
import dotenv
import datetime
from urllib import request
import pandas as pd
import numpy as np

# =============== PARTIE 1 =============
# =============== 1.1 : REDDIT ===============
# Library
import praw

# =============== 1.2 : ArXiv ===============
# Libraries
import xmltodict


# TODO: Translate if needed.
# Load Environment Variables
dotenv.load_dotenv()

reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
reddit_user_agent = os.getenv("REDDIT_USER_AGENT")

# Reddit

reddit = praw.Reddit(
    client_id=reddit_client_id,
    client_secret=reddit_client_secret,
    user_agent=reddit_user_agent,
)

# TODO: Translate print
print("Fetching data from reddit ...")
subr = reddit.subreddit("Coronavirus").hot(limit=100)
textes_Reddit = []

for post in subr:
    texte = post.title
    texte = texte.replace("\n", " ")
    # TODO: Translate source
    data = {"texte": texte, "source": "reddit"}
    textes_Reddit.append(data)

# TODO: Translate print
print("Reddit data downloaded.")

textes_Arxiv = []

query = "covid"
url = (
    "http://export.arxiv.org/api/query?search_query=all:"
    + query
    + "&start=0&max_results=100"
)
# TODO: Translate print
print("Fetching data from arxiv ...")
url_read = request.urlopen(url).read()

# url_read est un "byte stream" qui a besoin d'être décodé
data = url_read.decode()
dico = xmltodict.parse(data)  # xmltodict permet d'obtenir un objet ~JSON
docs = dico["feed"]["entry"]
for d in docs:
    texte = d["title"] + ". " + d["summary"]
    texte = texte.replace("\n", " ")
    data = {"texte": texte, "source": "arxiv"}
    textes_Arxiv.append(data)
# TODO: Translate print
print("Arxiv data downloaded.")
# on concatène tout ça :

df_data = textes_Reddit + textes_Arxiv

# TODO: Translate
# Create a Backup file.
print("Saving Data to output.csv")
df = pd.DataFrame(df_data)
df["id"] = range(1, len(df) + 1)
with open("./data/output.csv", mode="w", encoding="utf-8") as file:
    df.to_csv(file, sep="\t", index=False)

# TODO: Translate
# Load Data

print("Loading Data from output.csv")
with open("./data/output.csv", mode="r+", encoding="utf-8") as file:
    corpus = pd.read_csv(file, sep="\t")

    print("Longueur du corpus : " + str(len(corpus)))

    indices_to_drop = []
    for index, doc in corpus.iterrows():
        # nombre de phrases
        print("*****************************")
        num_de_phrases = len(doc["texte"].split("."))
        num_de_mots = len(doc["texte"].split(" "))
        print(f"Nombre de phrases : {num_de_phrases}")
        print(f"Nombre de mots : {num_de_mots}")

        if num_de_mots < 20:
            indices_to_drop.append(index)

    # TODO: Translate
    print("Deleting less than 20 Chars docs ...")
    corpus.drop(indices_to_drop, inplace=True)
    corpus.to_csv(file, index=False)
    print(f"{len(indices_to_drop)} Docs has been deleted.")

    nb_phrases = [len(doc["texte"].split(".")) for _, doc in corpus.iterrows()]
    print(f"Moyenne du nombre de phrases : {np.mean(nb_phrases)}")

    nb_mots = [len(doc["texte"].split(" ")) for _, doc in corpus.iterrows()]
    print(f"Moyenne du nombre de mots : {np.mean(nb_mots)}")

    print(f"Nombre total de mots dans le corpus : {np.sum(nb_mots)}")


aujourdhui = datetime.datetime.now()
print(aujourdhui)
