import os
import dotenv
import datetime

# Correction de G. Poux-Médard, 2021-2022

# =============== PARTIE 1 =============
# =============== 1.1 : REDDIT ===============
# Library

import praw

# =============== 1.2 : ArXiv ===============
# Libraries
import urllib
import xmltodict

import pandas as pd

dotenv.load_dotenv()


def get_data_from_reddit(max_results):
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

    # Requête
    hot_posts = reddit.subreddit("Coronavirus").hot(
        limit=max_results
    )  # .top("all", limit=limit)#

    # Récupération du texte
    docs_bruts = []
    for i, post in enumerate(hot_posts):
        if i % 10 == 0:
            print("Reddit:", i, "/", max_results)

        titre = post.title.replace("\n", "")
        auteur = str(post.author)
        date = datetime.datetime.fromtimestamp(post.created).strftime("%Y/%m/%d")
        url = "https://www.reddit.com/" + post.permalink
        texte = post.selftext.replace("\n", " ")
        num_comments = len(post.comments.list())

        entry = {
            "titre": titre,
            "authors": auteur,
            "date": date,
            "texte": texte,
            "num_comments": num_comments,
            "url": url,
            "source": "Reddit",
        }
        docs_bruts.append(entry)
        
    return docs_bruts


def get_data_from_arxiv(max_results):
    # Paramètres
    query_terms = ["clustering", "Dirichlet"]

    # Requête
    url = f'http://export.arxiv.org/api/query?search_query=all:{"+".join(query_terms)}&start=0&max_results={max_results}'
    data = urllib.request.urlopen(url)

    # Format dict (OrderedDict)
    data = xmltodict.parse(data.read().decode("utf-8"))

    # Ajout résumés à la liste
    docs_bruts = []
    for i, entry in enumerate(data["feed"]["entry"]):
        if i % 10 == 0:
            print("ArXiv:", i, "/", max_results)
        titre = entry["title"].replace("\n", "")
        date = datetime.datetime.strptime(
            entry["published"], "%Y-%m-%dT%H:%M:%SZ"
        ).strftime(
            "%Y/%m/%d"
        )  # Formatage de la date en année/mois/jour avec librairie datetime
        summary = entry["summary"].replace(
            "\n", " "
        )  # On enlève les retours à la ligne
        if isinstance(entry["author"], dict):
            authors = entry["author"]["name"]
        if isinstance(entry["author"], list):
            authors = [
                a["name"] for a in entry["author"]
            ]  # On fait une liste d'auteurs, séparés par une virgule
            authors = ",".join(authors)
        url = entry["id"]
        entry = {
            "titre": titre,
            "authors": authors,
            "date": date,
            "url": url,
            "summary": summary,
            "source": "ArXiv",
        }

        docs_bruts.append(entry)
    return docs_bruts


def download_data(max_reddit_results=10, max_arxiv_results=10):
    df_data = [
        *get_data_from_reddit(max_results=max_reddit_results),
        *get_data_from_arxiv(max_results=max_arxiv_results),
    ]
    df = pd.DataFrame(df_data)
    df["id"] = range(1, len(df) + 1)
    with open("./data/data.csv", mode="wb") as file:
        df.to_csv(file, sep="\t", index=False)


def load_data():
    if not os.path.exists("./data/data.csv"):
        download_data()

    data = None
    with open("./data/data.csv", mode="rb") as file:
        data = pd.read_csv(file, sep="\t")
    return data


if __name__ == "__main__":
    download_data()
