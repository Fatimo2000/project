from data.download_data import load_data

# =============== PARTIE 2 =============
# =============== 2.1, 2.2 : CLASSE DOCUMENT ===============
from Document import DocumentFactory

# =============== 2.3 : MANIPS ===============
import datetime
from tqdm import tqdm

# =============== 2.4, 2.5 : CLASSE AUTEURS ===============
from Classes import Author

# =============== 2.7, 2.8 : CORPUS ===============
# from Corpus import Corpus
from Corpus import Corpus

# =============== 2.9 : SAUVEGARDE ===============
import pickle

data = load_data()
# print(data)

docs = []
for _, doc in data.iterrows():
    texte = doc.get("summary") or doc.get("texte")
    if isinstance(texte, str):
        docs.append(texte.replace("\n", " "))
# =============== 1.3 : Exploitation ===============
print(f"# docs avec doublons : {len(docs)}")
docs = list(set(docs))
print(f"# docs sans doublons : {len(docs)}")

for i, doc in enumerate(docs):
    print(
        f"Document {i}\t# caractères : {len(doc)}\t# mots : {len(doc.split(' '))}\t# phrases : {len(doc.split('.'))}"
    )
    if len(doc) < 100:
        docs.remove(doc)

collection = []
factory = DocumentFactory()
for _, post in data.iterrows():
    if (
        post.source == "ArXiv"
    ):  # Les fichiers de ArXiv ou de Reddit sont pas formatés de la même manière à ce stade.
        # doc_classe = ArxivDocument(titre, authors, date, doc["id"], summary)  # Création du Document
        authors = post["authors"].split(",")
        doc_classe = factory.create_arxiv_document(
            titre=post["titre"],
            authors=authors,
            date=post["date"],
            url=post["url"],
            summary=post["summary"],
        )
        collection.append(doc_classe)  # Ajout du Document à la liste.

    elif post.source == "Reddit":
        # doc_classe = RedditDocument(titre, auteur, date, url, texte, num_comments=num_comments)
        doc_classe = factory.create_reddit_document(
            titre=post["titre"],
            auteur=post["authors"],
            date=post["date"],
            url=post["url"],
            texte=post["texte"],
            num_comments=post["num_comments"],
        )
        collection.append(doc_classe)

# Création de l'index de documents
id2doc = {}
for i, doc in enumerate(collection):
    id2doc[i] = doc.titre


# =============== 2.6 : DICT AUTEURS ===============
authors = {}
aut2id = {}
num_auteurs_vus = 0

# Création de la liste+index des Auteurs
for doc in collection:
    if doc.auteur not in aut2id:
        num_auteurs_vus += 1
        authors[num_auteurs_vus] = Author(doc.auteur)
        aut2id[doc.auteur] = num_auteurs_vus

    authors[aut2id[doc.auteur]].add(doc.texte)


corpus = Corpus("Mon corpus")

# Construction du corpus à partir des documents
for doc in collection:
    corpus.add(doc)
# corpus.show(tri="abc")
# print(repr(corpus))


# Ouverture d'un fichier, puis écriture avec pickle
with open("./data/corpus.pkl", "wb") as f:
    pickle.dump(corpus, f)

# Supression de la variable "corpus"
del corpus

# Ouverture du fichier, puis lecture avec pickle
with open("./data/corpus.pkl", "rb") as f:
    corpus = pickle.load(f)

# La variable est réapparue
print(corpus.show())
