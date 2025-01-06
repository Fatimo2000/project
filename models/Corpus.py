import os
from collections import defaultdict
from typing import Union
import pandas as pd
from models.Classes import Author
from models.Document import ArxivDocument, RedditDocument
import pickle
import re


# =============== 2.7 : CLASSE CORPUS ===============
class Corpus:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Corpus, cls).__new__(cls)
        return cls._instance

    def __init__(self, nom):
        if not hasattr(self, "initialized"):
            self.initialized = True
            self.nom = nom
        self.concatenated_string = None
        self.authors = {}
        self.aut2id = {}
        self.id2doc = {}
        self.ndoc = 0
        self.naut = 0

    def add(self, doc: Union[RedditDocument, ArxivDocument]):
        if self.concatenated_string is not None:
            self.concatenated_string = None

        if doc.auteur not in self.aut2id:
            self.naut += 1
            self.authors[self.naut] = Author(doc.auteur)
            self.aut2id[doc.auteur] = self.naut
        self.authors[self.aut2id[doc.auteur]].add(doc.texte)

        self.ndoc += 1
        self.id2doc[self.ndoc] = doc

    # =============== 2.8 : REPRESENTATION ===============
    def show(self, n_docs=-1, tri="abc"):
        docs = list(self.id2doc.values())
        if tri == "abc":  # Tri alphabétique
            docs = list(sorted(docs, key=lambda x: x.titre.lower()))[:n_docs]
        elif tri == "123":  # Tri temporel
            docs = list(sorted(docs, key=lambda x: x.date))[:n_docs]

        print("\n".join(list(map(repr, docs))))

    def save(self, file_loc):
        try:
            with open(file_loc, "wb") as f:
                pickle.dump(self, f)
                print(f"Corpus saved to {file_loc}.")
        except Exception as e:
            print(f"An error occurred while saving the corpus: {e}")

    @staticmethod
    def load(file_loc):
        if not os.path.exists(file_loc):
            print(f"File not found: {file_loc}")
            return None
        try:
            with open(file_loc, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"An error occurred while loading the corpus: {e}")
            return None

    def create_concatenated_string(self):
        string = "".join(
            doc.texte for _, doc in self.id2doc.items() if isinstance(doc.texte, str)
        )
        # for _, doc in self.id2doc.items():
        #     string.join(doc.texte if isinstance(doc.texte, str) else "")
        return string

    def search(self, keyword: str):
        if self.concatenated_string is None:
            self.concatenated_string = self.create_concatenated_string()
        pattern = re.compile(keyword)
        matches = pattern.findall(self.concatenated_string)
        return matches if matches else None

    def concorde(self, expression: str, context_size: int = 30):
        if self.concatenated_string is None:
            self.concatenated_string = self.create_concatenated_string()
        pattern = re.compile(expression)
        matches = pattern.finditer(self.concatenated_string)

        results = []
        for match in matches:
            start = match.start()
            end = match.end()
            context_start = max(0, start - context_size)
            context_end = min(len(self.concatenated_string), end + context_size)
            context = self.concatenated_string[context_start:context_end]
            results.append({"match": match.group(), "context": context})

        df = pd.DataFrame(results)
        return df if not df.empty else None

    @staticmethod
    def clean_text(text: str):
        if not isinstance(text, str):
            return ""
        clean_text = text.lower()
        clean_text = text.replace("\n", " ")
        clean_text = re.sub(r"[^\w\s]", "", text)
        clean_text = re.sub(r"\d+", "", text)
        return clean_text

    def build_vocabulary(self):
        word_count = defaultdict(int)  # To count total occurrences of each word
        doc_count = defaultdict(int)  # To count document frequency of each word

        for _, doc in self.id2doc.items():
            cleaned_doc = self.clean_text(doc.texte)

            words = cleaned_doc.split()  # Split by whitespace

            # Use a set to keep track of words in the current document
            unique_words = set(words)

            # Count occurrences of each word
            for word in words:
                word_count[word] += 1

            # Count document frequency
            for word in unique_words:
                doc_count[word] += 1

        # Create a DataFrame for frequency table
        freq_table = pd.DataFrame(
            {
                "word": word_count.keys(),
                "term_frequency": word_count.values(),
                "document_frequency": [doc_count[word] for word in word_count.keys()],
            }
        )

        return freq_table

    def stats(self, n: int):
        freq_table = self.build_vocabulary()
        most_frequent_words = freq_table.nlargest(n, "term_frequency")
        print(f"The {n} most frequent words:\n", most_frequent_words)

    def __repr__(self):
        docs = list(self.id2doc.values())
        docs = list(sorted(docs, key=lambda x: x.titre.lower()))

        return "\n".join(list(map(str, docs)))
