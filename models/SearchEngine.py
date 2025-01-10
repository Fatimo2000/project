from collections import defaultdict
from tqdm import tqdm
from models import Corpus, Document
from scipy import sparse
import pandas as pd
import numpy as np


class SearchEngine:
    """A search engine that allows querying a corpus of documents."""

    def __init__(self, corpus: Corpus.Corpus):
        """Initializes the SearchEngine with a Corpus object.

        Args:
            corpus (Corpus): An instance of the Corpus class.
        """
        self.corpus: Corpus.Corpus = corpus  # Store the corpus
        self.vocab = self.build_vocabulary()  # Build vocabulary from the corpus
        self.mat_tf = self.build_term_frequency_matrix()  # Build term frequency matrix
        self.mat_tfxidf = self.build_tfidf_matrix()  # Build TF-IDF matrix

    def build_vocabulary(self) -> dict:
        """Builds a vocabulary dictionary from the documents.

        Returns:
            dict: A dictionary containing words and their information, such as total occurrences and document frequency.
        """
        word_count = defaultdict(int)  # To count total occurrences of each word
        doc_count = defaultdict(int)  # To count document frequency of each word

        # Iterate through each document in the corpus
        for _, doc in self.corpus.id2doc.items():
            cleaned_doc = self.corpus.clean_text(doc.texte)  # Clean the document text
            words = cleaned_doc.split()  # Split into words
            unique_words = set(words)  # Use a set to avoid duplicates

            # Count occurrences of each word
            for word in words:
                word_count[word] += 1

            # Count document frequency for unique words
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
        print("SearchEngine Vocabulary Built.")
        return vocab

    def build_term_frequency_matrix(self) -> sparse.csr_matrix:
        """Builds the term frequency matrix (TF) for the documents.

        Returns:
            csr_matrix: A sparse matrix representing term frequencies.
        """
        num_docs = self.corpus.ndoc  # Total number of documents
        num_words = len(self.vocab)  # Total number of unique words

        # Initialize lists for sparse matrix construction
        data = []
        row_indices = []
        col_indices = []

        # Iterate through each document to populate the sparse matrix
        for doc_id, doc in self.corpus.id2doc.items():
            cleaned_doc = self.corpus.clean_text(doc.texte)  # Clean the document text
            words = cleaned_doc.split()  # Split into words
            word_count = defaultdict(
                int
            )  # To count occurrences of each word in the document

            # Count occurrences of each word in the document
            for word in words:
                if word in self.vocab:
                    word_count[word] += 1

            # Fill the sparse matrix with word counts
            for word, count in word_count.items():
                word_index = self.vocab[word]["id"]  # Get the index of the word
                data.append(count)  # Append the count
                row_indices.append(doc_id - 1)  # Adjusting for zero-based index
                col_indices.append(word_index)  # Append the word index

        # Create the sparse matrix from the collected data
        mat = sparse.csr_matrix(
            (data, (row_indices, col_indices)), shape=(num_docs, num_words)
        )
        print("SearchEngine Term Frequency Matrix built.")
        return mat

    def build_tfidf_matrix(self) -> sparse.csr_matrix:
        """Builds the TF-IDF matrix from the term frequency matrix.

        Returns:
            csr_matrix: A sparse matrix representing TF-IDF values.
        """
        num_docs = self.corpus.ndoc  # Total number of documents

        # Get the term frequency matrix
        tf_matrix: sparse.csr_matrix = self.mat_tf

        # Calculate the document frequency for each word
        doc_frequencies = np.array(
            [self.vocab[word]["document_frequency"] for word in self.vocab]
        )

        # Calculate the Inverse Document Frequency (IDF)
        idf = np.log(num_docs / (1 + doc_frequencies))

        # Create a diagonal matrix for IDF values
        idf_matrix = sparse.diags(idf)

        # Calculate the TF-IDF matrix by multiplying TF matrix with IDF matrix
        tfxidf_matrix = tf_matrix.dot(idf_matrix)

        print("SearchEngine TfIdf Matrix Built.")
        return tfxidf_matrix

    def vectorize_query(self, keywords: str) -> np.ndarray:
        """Transforms the input keywords into a vector based on the vocabulary.

        Args:
            keywords (str): The keywords to vectorize.

        Returns:
            np.ndarray: A vector representation of the keywords.
        """
        keyword_vector = np.zeros(len(self.vocab))  # Initialize a zero vector
        cleaned_keywords = self.corpus.clean_text(
            keywords
        ).split()  # Clean and split keywords

        # Increment the count for each keyword in the vector
        for word in cleaned_keywords:
            if word in self.vocab:
                keyword_vector[self.vocab[word]["id"]] += (
                    1  # Increment the count for the word
                )

        return keyword_vector

    def cosine_similarity(
        self, query_vector: np.ndarray, document_vector: sparse.csr_matrix
    ) -> float:
        """Calculates the cosine similarity between the query vector and a document vector.

        Args:
            query_vector (np.ndarray): The vector representation of the query.
            document_vector (csr_matrix): The document vector to compare against.

        Returns:
            float: The cosine similarity score.
        """
        dot_product = np.dot(
            query_vector, document_vector.toarray().flatten()
        )  # Calculate dot product
        norm_query = np.linalg.norm(
            query_vector
        )  # Calculate the norm of the query vector
        norm_document = np.linalg.norm(
            document_vector.toarray().flatten()
        )  # Calculate the norm of the document vector

        # Return 0 if either vector has zero length to avoid division by zero
        if norm_query == 0 or norm_document == 0:
            return 0.0
        return dot_product / (
            norm_query * norm_document
        )  # Return cosine similarity score

    def search(self, keywords: str, top_n: int) -> pd.DataFrame:
        """Searches for the given keywords in the corpus and returns the top_n documents.

        Args:
            keywords (str): The keywords to search for, separated by spaces.
            top_n (int): The number of top documents to return.

        Returns:
            DataFrame: A pandas DataFrame containing the search results.
        """
        # Vectorize the query
        query_vector = self.vectorize_query(keywords)

        # Create a dictionary to hold document scores
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
            scores[doc_id] = self.cosine_similarity(
                query_vector, document_vector
            )  # Calculate similarity score

        # Sort the documents based on scores in descending order
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        # Prepare the results DataFrame
        results = []
        for doc_id, score in sorted_docs[:top_n]:
            document: Document.Document = self.corpus.id2doc[
                doc_id + 1
            ]  # Retrieve the document
            results.append(
                {
                    "document_index": doc_id + 1,  # Adjusting for one-based index
                    "score": score,  # Document score
                    "document": document,  # Document object
                    "author": document.auteur,  # Document author
                }
            )

        return pd.DataFrame(results)  # Return the results as a DataFrame

    def search_with_filters(
        self,
        keywords: str,
        top_n: int,
        authors=None,
        doc_type=None,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """Searches for the given keywords in the corpus and returns the top_n documents with optional filters.

        Args:
            keywords (str): The keywords to search for, separated by spaces.
            top_n (int): The number of top documents to return.
            authors (list, optional): Optional list of authors to filter by.
            doc_type (str, optional): Optional document type to filter by.
            start_date (datetime, optional): Optional start date for filtering documents.
            end_date (datetime, optional): Optional end date for filtering documents.

        Returns:
            DataFrame: A pandas DataFrame containing the search results.
        """
        # Query the corpus for documents matching the criteria
        filtered_docs = self.corpus.query_documents(
            keywords=keywords.split(),
            authors=authors,
            doc_type=doc_type,
            start_date=start_date,
            end_date=end_date,
        )

        # If no documents match the criteria, return an empty DataFrame
        if not filtered_docs:
            return pd.DataFrame(
                columns=["document_index", "score", "document", "author"]
            )

        # Vectorize the query
        query_vector = self.vectorize_query(keywords)

        # Create a dictionary to hold document scores
        scores = defaultdict(float)

        # Calculate similarity scores for each filtered document
        for doc_id, document in filtered_docs.items():
            document_vector = self.mat_tfxidf[
                doc_id - 1, :
            ]  # Adjust for zero-based index
            scores[doc_id] = self.cosine_similarity(
                query_vector, document_vector
            )  # Calculate similarity score

        # Sort the documents based on scores in descending order
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        # Prepare the results DataFrame
        results = []
        for doc_id, score in sorted_docs[:top_n]:
            document: Document.Document = self.corpus.id2doc[
                doc_id
            ]  # Retrieve the document
            results.append(
                {
                    "document_index": doc_id,  # Document index
                    "score": score,  # Document score
                    "document": document,  # Document object
                    "author": document.auteur,  # Document author
                }
            )

        return pd.DataFrame(results)  # Return the results as a DataFrame
