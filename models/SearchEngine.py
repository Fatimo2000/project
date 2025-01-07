from collections import defaultdict
from tqdm import tqdm
from models import Corpus, Document
from scipy import sparse
import pandas as pd
import numpy as np


class SearchEngine:
    def __init__(self, corpus: Corpus.Corpus):
        """
        Initializes the SearchEngine with a Corpus object.

        Parameters:
        - corpus (Corpus): An instance of the Corpus class.
        """
        self.corpus: Corpus.Corpus = corpus
        self.vocab = self.build_vocabulary()
        self.mat_tf = self.build_term_frequency_matrix()
        self.mat_tfxidf = self.build_tfidf_matrix()

    def build_vocabulary(self) -> dict:
        """
        Builds a vocabulary dictionary from the documents.

        Returns:
        - dict: A dictionary containing words and their information.
        """
        word_count = defaultdict(int)  # To count total occurrences of each word
        doc_count = defaultdict(int)  # To count document frequency of each word

        for _, doc in self.corpus.id2doc.items():
            cleaned_doc = self.corpus.clean_text(doc.texte)
            words = cleaned_doc.split()
            unique_words = set(words)  # Use a set to avoid duplicates

            # Count occurrences of each word
            for word in words:
                word_count[word] += 1

            # Count document frequency
            for word in unique_words:
                doc_count[word] += 1

        # Create the vocabulary dictionary
        vocab = {
            word: {
                "id": idx,
                "total_occurrences": word_count[word],
                "document_frequency": doc_count[word],
            }
            for idx, word in enumerate(sorted(word_count.keys()))
        }
        print("SearchEnginge Vocabulary Built.")
        return vocab

    def build_term_frequency_matrix(self) -> sparse.csr_matrix:
        """
        Builds the term frequency matrix (TF) for the documents.

        Returns:
        - csr_matrix: A sparse matrix representing term frequencies.
        """
        num_docs = self.corpus.ndoc
        num_words = len(self.vocab)

        # Initialize a sparse matrix
        data = []
        row_indices = []
        col_indices = []

        for doc_id, doc in self.corpus.id2doc.items():
            cleaned_doc = self.corpus.clean_text(doc.texte)
            words = cleaned_doc.split()
            word_count = defaultdict(int)

            # Count occurrences of each word in the document
            for word in words:
                if word in self.vocab:
                    word_count[word] += 1

            # Fill the sparse matrix
            for word, count in word_count.items():
                word_index = self.vocab[word]["id"]
                data.append(count)
                row_indices.append(doc_id - 1)  # Adjusting for zero-based index
                col_indices.append(word_index)

        # Create the sparse matrix
        mat = sparse.csr_matrix(
            (data, (row_indices, col_indices)), shape=(num_docs, num_words)
        )
        print("SearchEnginge Term Frequency Matrix built.")
        return mat

    def build_tfidf_matrix(self) -> sparse.csr_matrix:
        """
        Builds the TF-IDF matrix from the term frequency matrix.

        Returns:
        - csr_matrix: A sparse matrix representing TF-IDF values.
        """
        num_docs = self.corpus.ndoc

        # mat_tf = self.build_term_frequency_matrix()
        # Get the term frequency matrix
        tf_matrix: sparse.csr_matrix = self.mat_tf

        # Calculate the document frequency for each word
        doc_frequencies = np.array(
            [self.vocab[word]["document_frequency"] for word in self.vocab]
        )

        idf = np.log(num_docs / (1 + doc_frequencies))

        # Create a diagonal matrix for IDF values
        idf_matrix = sparse.diags(idf)

        # Calculate the TF-IDF matrix
        tfxidf_matrix = tf_matrix.dot(idf_matrix)

        print("SearchEngine TfIdf Matrix Built.")
        return tfxidf_matrix

    def vectorize_query(self, keywords: str) -> np.ndarray:
        """
        Transforms the input keywords into a vector based on the vocabulary.

        Parameters:
        - keywords (str): The keywords to vectorize.

        Returns:
        - np.ndarray: A vector representation of the keywords.
        """
        keyword_vector = np.zeros(len(self.vocab))
        cleaned_keywords = self.corpus.clean_text(keywords).split()

        for word in cleaned_keywords:
            if word in self.vocab:
                keyword_vector[self.vocab[word]["id"]] += (
                    1  # Increment the count for the word
                )

        return keyword_vector

    def cosine_similarity(
        self, query_vector: np.ndarray, document_vector: sparse.csr_matrix
    ) -> float:
        """
        Calculates the cosine similarity between the query vector and a document vector.

        Parameters:
        - query_vector (np.ndarray): The vector representation of the query.
        - document_vector (csr_matrix): The document vector to compare against.

        Returns:
        - float: The cosine similarity score.
        """
        dot_product = np.dot(query_vector, document_vector.toarray().flatten())
        norm_query = np.linalg.norm(query_vector)
        norm_document = np.linalg.norm(document_vector.toarray().flatten())
        if norm_query == 0 or norm_document == 0:
            return 0.0
        return dot_product / (norm_query * norm_document)

    def search(self, keywords: str, top_n: int) -> pd.DataFrame:
        """
        Searches for the given keywords in the corpus and returns the top_n documents.

        Parameters:
        - keywords (str): The keywords to search for, separated by spaces.
        - top_n (int): The number of top documents to return.

        Returns:
        - DataFrame: A pandas DataFrame containing the search results.
        """
        # Vectorize the query
        query_vector = self.vectorize_query(keywords)

        # Create a list to hold document scores
        scores = defaultdict(float)

        # Calculate similarity scores for each document
        for doc_id in tqdm(
            range(self.corpus.ndoc),
            desc="Searching ...",
            unit="Docs",
        ):
            document_vector = self.mat_tfxidf[
                doc_id, :
            ]  # Get the TF-IDF vector for the document
            scores[doc_id] = self.cosine_similarity(query_vector, document_vector)

        # Sort the documents based on scores
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        # Prepare the results DataFrame
        results = []
        for doc_id, score in sorted_docs[:top_n]:
            document: Document.Document = self.corpus.id2doc[doc_id + 1]
            results.append(
                {
                    "document_index": doc_id + 1,  # Adjusting for one-based index
                    "score": score,
                    "document": document,  # Adjusting for one-based index
                    "author": document.auteur,
                }
            )

        return pd.DataFrame(results)
