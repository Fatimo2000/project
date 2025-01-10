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
    """Represents a collection of documents with functionalities for management and analysis."""

    def __init__(self, nom):
        """Initializes the Corpus object.

        Args:
            nom (str): The name of the corpus.
        """
        self.nom = nom  # Name of the corpus
        self.concatenated_string = None  # Placeholder for concatenated document texts
        self.authors = {}  # Dictionary to hold authors
        self.aut2id = {}  # Mapping from author names to IDs
        self.id2doc = {}  # Mapping from document IDs to documents
        self.ndoc = 0  # Number of documents in the corpus
        self.naut = 0  # Number of authors in the corpus

    def add(self, doc: Union[RedditDocument, ArxivDocument]):
        """Adds a document to the corpus.

        Args:
            doc (Union[RedditDocument, ArxivDocument]): The document to be added.
        """
        # Reset concatenated string if it exists
        if self.concatenated_string is not None:
            self.concatenated_string = None

        # Add the author if not already present
        if doc.auteur not in self.aut2id:
            self.naut += 1  # Increment author count
            self.authors[self.naut] = Author(doc.auteur)  # Create new Author object
            self.aut2id[doc.auteur] = self.naut  # Map author name to ID

        # Add document text to the author's collection
        self.authors[self.aut2id[doc.auteur]].add(doc.texte)

        # Increment document count and map document ID to the document
        self.ndoc += 1
        self.id2doc[self.ndoc] = doc

    # =============== 2.8 : REPRESENTATION ===============
    def show(self, n_docs=-1, tri="abc"):
        """Displays documents in the corpus.

        Args:
            n_docs (int, optional): Number of documents to display (default is -1, which shows all).
            tri (str, optional): Sorting method ("abc" for alphabetical, "123" for chronological).
        """
        docs = list(self.id2doc.values())  # Get all documents
        if tri == "abc":  # Sort alphabetically by title
            docs = list(sorted(docs, key=lambda x: x.titre.lower()))[:n_docs]
        elif tri == "123":  # Sort chronologically by date
            docs = list(sorted(docs, key=lambda x: x.date))[:n_docs]

        # Print the string representation of the documents
        print("\n".join(list(map(repr, docs))))

    def save(self, file_loc):
        """Saves the corpus to a file using pickle.

        Args:
            file_loc (str): The location where the corpus will be saved.
        """
        try:
            with open(file_loc, "wb") as f:
                pickle.dump(self, f)  # Serialize the corpus object
                print(f"Corpus saved to {file_loc}.")
        except Exception as e:
            print(f"An error occurred while saving the corpus: {e}")

    @staticmethod
    def load(file_loc):
        """Loads a corpus from a file.

        Args:
            file_loc (str): The location of the file to load.

        Returns:
            Corpus or None: The loaded Corpus object or None if loading fails.
        """
        if not os.path.exists(file_loc):
            print(f"File not found: {file_loc}")
            return None
        try:
            with open(file_loc, "rb") as f:
                return pickle.load(f)  # Deserialize the corpus object
        except Exception as e:
            print(f"An error occurred while loading the corpus: {e}")
            return None

    def create_concatenated_string(self):
        """Creates a single concatenated string of all document texts.

        Returns:
            str: A string containing all document texts.
        """
        string = "".join(
            doc.texte for _, doc in self.id2doc.items() if isinstance(doc.texte, str)
        )
        return string

    def search(self, keyword: str):
        """Searches for occurrences of a keyword in the concatenated string.

        Args:
            keyword (str): The keyword to search for.

        Returns:
            list or None: A list of matches or None if no matches are found.
        """
        if self.concatenated_string is None:
            self.concatenated_string = (
                self.create_concatenated_string()
            )  # Create if not present
        pattern = re.compile(keyword)  # Compile the keyword into a regex pattern
        matches = pattern.findall(self.concatenated_string)  # Find matches
        return matches if matches else None  # Return matches or None

    def concorde(self, expression: str, context_size: int = 30):
        """Finds occurrences of a regular expression and provides context.

        Args:
            expression (str): The regex expression to search for.
            context_size (int, optional): The number of characters to include as context around matches.

        Returns:
            pd.DataFrame or None: A DataFrame containing matches and their context.
        """
        if self.concatenated_string is None:
            self.concatenated_string = (
                self.create_concatenated_string()
            )  # Create if not present
        pattern = re.compile(expression)  # Compile the regex expression
        matches = pattern.finditer(self.concatenated_string)  # Find all matches

        results = []  # List to hold results with context
        for match in matches:
            start = match.start()  # Start index of the match
            end = match.end()  # End index of the match
            context_start = max(0, start - context_size)  # Start index for context
            context_end = min(
                len(self.concatenated_string), end + context_size
            )  # End index for context
            context = self.concatenated_string[
                context_start:context_end
            ]  # Extract context
            results.append(
                {"match": match.group(), "context": context}
            )  # Append result

        df = pd.DataFrame(results)  # Convert results to a DataFrame
        return df if not df.empty else None  # Return DataFrame or None if empty

    @staticmethod
    def clean_text(text: str):
        """Cleans text by converting to lowercase and removing unwanted characters.

        Args:
            text (str): The text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        if not isinstance(text, str):
            return ""  # Return empty string if input is not a string
        clean_text = text.lower()  # Convert to lowercase
        clean_text = text.replace("\n", " ")  # Replace newlines with spaces
        clean_text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        clean_text = re.sub(r"\d+", "", text)  # Remove digits
        return clean_text

    def get_documents_by_type(self, doc_type: str):
        """Retrieves documents of a specific type.

        Args:
            doc_type (str): The type of documents to retrieve.

        Returns:
            list: A list of documents of the specified type.
        """
        return [doc for doc in self.id2doc.values() if doc.getType() == doc_type]

    def build_vocabulary(self):
        """Builds a vocabulary table with term and document frequencies.

        Returns:
            pd.DataFrame: A DataFrame containing words, term frequencies, and document frequencies.
        """
        word_count = defaultdict(int)  # To count total occurrences of each word
        doc_count = defaultdict(int)  # To count document frequency of each word

        for _, doc in self.id2doc.items():
            cleaned_doc = self.clean_text(doc.texte)  # Clean the document text

            words = cleaned_doc.split()  # Split the cleaned text into words

            # Use a set to keep track of unique words in the current document
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

        return freq_table  # Return the frequency table

    def stats(self, n: int):
        """Displays the most frequent words in the corpus.

        Args:
            n (int): The number of most frequent words to display.
        """
        freq_table = self.build_vocabulary()  # Build the vocabulary
        most_frequent_words = freq_table.nlargest(
            n, "term_frequency"
        )  # Get top n words
        print(f"The {n} most frequent words:\n", most_frequent_words)

    def group_by_time_period(self, period="year"):
        """Groups documents by a specified time period and returns a frequency count of words.

        Args:
            period (str, optional): The time period for grouping (e.g., 'year', 'month').

        Returns:
            pd.DataFrame: A DataFrame with time periods and word frequencies.
        """
        time_grouped = defaultdict(
            lambda: defaultdict(int)
        )  # Nested dictionary for word counts by time

        for _, doc in self.id2doc.items():
            # Ensure the date is a datetime object
            if isinstance(doc.date, str):
                doc.date = datetime.strptime(
                    doc.date, "%Y/%m/%d"
                )  # Adjust format as needed

            # Determine the time period for grouping
            if period == "year":
                time_period = doc.date.year
            elif period == "month":
                time_period = f"{doc.date.year}-{doc.date.month:02d}"
            else:
                raise ValueError("Unsupported time period. Use 'year' or 'month'.")

            cleaned_doc = self.clean_text(doc.texte)  # Clean the document text
            words = cleaned_doc.split()  # Split the cleaned text into words
            for word in words:
                time_grouped[time_period][word] += (
                    1  # Count word occurrences for the time period
                )

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_grouped, orient="index").fillna(
            0
        )  # Create DataFrame and fill NaN with 0
        return df  # Return the grouped DataFrame

    def frequency_over_time(self, word):
        """Calculates the frequency of a specific word over time.

        Args:
            word (str): The word to track.

        Returns:
            pd.Series: Series containing time periods and frequencies of the word.
        """
        freq_df = self.group_by_time_period()  # Group documents by time
        word_freq = (
            freq_df[word]
            if word in freq_df.columns
            else pd.Series(0, index=freq_df.index)  # Return 0 if the word is not found
        )
        return word_freq  # Return the frequency of the word over time

    def calculate_tf_idf(self):
        """Calculates the TF-IDF for the words in the corpus.

        Returns:
            pd.DataFrame: DataFrame containing words and their TF-IDF scores.
        """
        word_count = defaultdict(int)  # To count total occurrences of each word
        doc_count = defaultdict(int)  # To count document frequency of each word
        total_docs = self.ndoc  # Total number of documents

        # Calculate term frequency (TF) and document frequency (DF)
        for _, doc in self.id2doc.items():
            cleaned_doc = self.clean_text(doc.texte)  # Clean the document text
            words = cleaned_doc.split()  # Split the cleaned text into words
            unique_words = set(words)  # Get unique words in the document

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
            tfidf_scores[word] = tf * idf  # Calculate TF-IDF score

        # Create a DataFrame for TF-IDF scores
        tfidf_df = pd.DataFrame(list(tfidf_scores.items()), columns=["word", "tfidf"])
        return tfidf_df.sort_values(
            by="tfidf", ascending=False
        )  # Return sorted TF-IDF DataFrame

    def query_documents(
        self, keywords=None, authors=None, doc_type=None, start_date=None, end_date=None
    ):
        """Queries the corpus for documents based on various criteria.

        Args:
            keywords (list, optional): A list of keywords to search for in the document texts.
            authors (list, optional): A list of authors to filter documents by.
            doc_type (str, optional): A specific document type to filter by (e.g., "Reddit" or "Arxiv").
            start_date (datetime, optional): The start date for filtering documents.
            end_date (datetime, optional): The end date for filtering documents.

        Returns:
            dict: A dictionary of documents that match the query criteria.
        """
        matching_docs = {}  # Dictionary to hold matching documents

        for doc_id, doc in self.id2doc.items():
            # Check author
            if authors and doc.auteur not in authors:
                continue  # Skip if author does not match

            # Check document type
            if doc_type and doc.getType() != doc_type:
                continue  # Skip if document type does not match

            # Check date range
            doc_date: datetime = doc.date
            doc_date = doc_date.date()  # Convert to date if necessary
            if start_date and doc_date < start_date:
                continue  # Skip if document date is before start date
            if end_date and doc_date > end_date:
                continue  # Skip if document date is after end date

            # Check keywords
            if keywords:
                cleaned_doc = self.clean_text(doc.texte)  # Clean the document text
                if not any(keyword.lower() in cleaned_doc for keyword in keywords):
                    continue  # Skip if none of the keywords are found

            # If all criteria are met, add the document to the results
            matching_docs[doc_id] = doc

        return matching_docs  # Return matching documents

    def calculate_bm25(self, query, k1=1.5, b=0.75):
        """Calculates the BM25 score for a query against the corpus.

        Args:
            query (list): A list of words to search for.
            k1 (float, optional): Term frequency saturation parameter.
            b (float, optional): Length normalization parameter.

        Returns:
            pd.DataFrame: A DataFrame containing words and their BM25 scores.
        """
        avg_doc_length = np.mean(
            [len(self.clean_text(doc.texte).split()) for doc in self.id2doc.values()]
        )  # Calculate average document length
        bm25_scores = defaultdict(float)  # Initialize BM25 scores

        for word in query:
            for _, doc in self.id2doc.items():
                cleaned_doc = self.clean_text(doc.texte)  # Clean the document text
                doc_length = len(cleaned_doc.split())  # Get document length
                term_freq = cleaned_doc.split().count(word)  # Count term frequency
                doc_freq = sum(
                    1
                    for d in self.id2doc.values()
                    if word in self.clean_text(d.texte).split()
                )  # Count documents containing the word

                # Calculate BM25 score
                if term_freq > 0:
                    idf = np.log(
                        (self.ndoc - doc_freq + 0.5) / (doc_freq + 0.5)
                    )  # Inverse document frequency
                    bm25_scores[word] += idf * (
                        (term_freq * (k1 + 1))
                        / (term_freq + k1 * (1 - b + b * (doc_length / avg_doc_length)))
                    )  # BM25 formula

        # Create a DataFrame for BM25 scores
        bm25_df = pd.DataFrame(list(bm25_scores.items()), columns=["word", "bm25"])
        return bm25_df.sort_values(
            by="bm25", ascending=False
        )  # Return sorted BM25 DataFrame

    def __repr__(self):
        """String representation of the Corpus object.

        Returns:
            str: A string containing the titles of all documents in the corpus.
        """
        docs = list(self.id2doc.values())  # Get all documents
        docs = list(
            sorted(docs, key=lambda x: x.titre.lower())
        )  # Sort documents by title

        return "\n".join(
            list(map(str, docs))
        )  # Return string representation of documents
