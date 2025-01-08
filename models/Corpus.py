import os
from collections import defaultdict
from datetime import datetime
from typing import Union
import pandas as pd
import numpy as np
from models.Classes import Author
from models.Document import ArxivDocument, RedditDocument
import pickle
import re


# =============== 2.7 : CLASSE CORPUS ===============
class Corpus:
    _instance = None

    def __init__(self, nom):
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

    def get_documents_by_type(self, doc_type: str):
        return [doc for doc in self.id2doc.values() if doc.getType() == doc_type]

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

    def group_by_time_period(self, period="year"):
        """
        Group documents by a specified time period and return a frequency count of words.

        Parameters:
        - period: The time period for grouping (e.g., 'year', 'month').

        Returns:
        - time_grouped: A DataFrame with time periods and word frequencies.
        """
        time_grouped = defaultdict(lambda: defaultdict(int))

        for _, doc in self.id2doc.items():
            # Ensure the date is a datetime object
            if isinstance(doc.date, str):
                doc.date = datetime.strptime(
                    doc.date, "%Y/%m/%d"
                )  # Adjust format as needed

            # Determine the time period
            if period == "year":
                time_period = doc.date.year
            elif period == "month":
                time_period = f"{doc.date.year}-{doc.date.month:02d}"
            else:
                raise ValueError("Unsupported time period. Use 'year' or 'month'.")

            cleaned_doc = self.clean_text(doc.texte)
            words = cleaned_doc.split()
            for word in words:
                time_grouped[time_period][word] += 1

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_grouped, orient="index").fillna(0)
        return df

    def frequency_over_time(self, word):
        """
        Calculate the frequency of a specific word over time.

        Parameters:
        - word: The word to track.

        Returns:
        - freq_df: DataFrame containing time periods and frequencies of the word.
        """
        freq_df = self.group_by_time_period()
        word_freq = (
            freq_df[word]
            if word in freq_df.columns
            else pd.Series(0, index=freq_df.index)
        )
        return word_freq

    def calculate_tf_idf(self):
        """
        Calculate the TF-IDF for the words in the corpus.

        Returns:
        - tfidf_df: DataFrame containing words and their TF-IDF scores.
        """
        word_count = defaultdict(int)  # To count total occurrences of each word
        doc_count = defaultdict(int)  # To count document frequency of each word
        total_docs = self.ndoc

        # Calculate term frequency (TF) and document frequency (DF)
        for _, doc in self.id2doc.items():
            cleaned_doc = self.clean_text(doc.texte)
            words = cleaned_doc.split()
            unique_words = set(words)

            # Count occurrences of each word
            for word in words:
                word_count[word] += 1

            # Count document frequency
            for word in unique_words:
                doc_count[word] += 1

        # Calculate TF-IDF
        tfidf_scores = {}
        for word, count in word_count.items():
            tf = count / sum(word_count.values())  # Term frequency
            idf = np.log(
                (total_docs + 1) / (doc_count[word] + 1)
            )  # Inverse document frequency with smoothing
            tfidf_scores[word] = tf * idf

        # Create a DataFrame for TF-IDF scores
        tfidf_df = pd.DataFrame(list(tfidf_scores.items()), columns=["word", "tfidf"])
        return tfidf_df.sort_values(by="tfidf", ascending=False)

    def query_documents(
        self, keywords=None, authors=None, doc_type=None, start_date=None, end_date=None
    ):
        """
        Query the corpus for documents based on various criteria.

        Parameters:
        - keywords: A list of keywords to search for in the document texts.
        - authors: A list of authors to filter documents by.
        - doc_type: A specific document type to filter by (e.g., "Reddit" or "Arxiv").
        - start_date: The start date for filtering documents (datetime object).
        - end_date: The end date for filtering documents (datetime object).

        Returns:
        - matching_docs: A list of documents that match the query criteria.
        """
        matching_docs = {}

        for doc_id, doc in self.id2doc.items():
            # Check author
            if authors and doc.auteur not in authors:
                continue

            # Check document type
            if doc_type and doc.getType() != doc_type:
                continue

            # Check date range
            doc_date: datetime = doc.date
            doc_date = doc_date.date()
            if start_date and doc_date < start_date:
                continue
            if end_date and doc_date > end_date:
                continue

            # Check keywords
            if keywords:
                cleaned_doc = self.clean_text(doc.texte)
                if not any(keyword.lower() in cleaned_doc for keyword in keywords):
                    continue

            # If all criteria are met, add the document to the results
            matching_docs[doc_id] = doc

        return matching_docs

    def calculate_bm25(self, query, k1=1.5, b=0.75):
        """
        Calculate the BM25 score for a query against the corpus.

        Parameters:
        - query: A list of words to search for.
        - k1: Term frequency saturation parameter.
        - b: Length normalization parameter.

        Returns:
        - bm25_scores: A DataFrame containing words and their BM25 scores.
        """
        avg_doc_length = np.mean(
            [len(self.clean_text(doc.texte).split()) for doc in self.id2doc.values()]
        )
        bm25_scores = defaultdict(float)

        for word in query:
            for _, doc in self.id2doc.items():
                cleaned_doc = self.clean_text(doc.texte)
                doc_length = len(cleaned_doc.split())
                term_freq = cleaned_doc.split().count(word)
                doc_freq = sum(
                    1
                    for d in self.id2doc.values()
                    if word in self.clean_text(d.texte).split()
                )

                # Calculate BM25 score
                if term_freq > 0:
                    idf = np.log(
                        (self.ndoc - doc_freq + 0.5) / (doc_freq + 0.5)
                    )  # Inverse document frequency
                    bm25_scores[word] += idf * (
                        (term_freq * (k1 + 1))
                        / (term_freq + k1 * (1 - b + b * (doc_length / avg_doc_length)))
                    )

        # Create a DataFrame for BM25 scores
        bm25_df = pd.DataFrame(list(bm25_scores.items()), columns=["word", "bm25"])
        return bm25_df.sort_values(by="bm25", ascending=False)

    def __repr__(self):
        docs = list(self.id2doc.values())
        docs = list(sorted(docs, key=lambda x: x.titre.lower()))

        return "\n".join(list(map(str, docs)))
