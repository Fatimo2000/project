from models.Classes import Document


class DocumentFactory:
    """A factory class for creating document instances."""

    @staticmethod
    def create_arxiv_document(titre="", authors=[""], date="", url="", summary=""):
        """Creates an instance of ArxivDocument.

        Args:
            titre (str, optional): The title of the document.
            authors (list, optional): A list of authors for the document.
            date (str, optional): The publication date of the document.
            url (str, optional): The URL of the document.
            summary (str, optional): A summary of the document.

        Returns:
            ArxivDocument: An instance of ArxivDocument.
        """
        return ArxivDocument(titre, authors, date, url, summary)

    @staticmethod
    def create_reddit_document(
        titre="", auteur="", date="", url="", texte="", num_comments=0
    ):
        """Creates an instance of RedditDocument.

        Args:
            titre (str, optional): The title of the document.
            auteur (str, optional): The author of the document.
            date (str, optional): The publication date of the document.
            url (str, optional): The URL of the document.
            texte (str, optional): The text content of the document.
            num_comments (int, optional): The number of comments on the document.

        Returns:
            RedditDocument: An instance of RedditDocument.
        """
        return RedditDocument(
            titre, auteur, date, url, texte, num_comments=num_comments
        )


class RedditDocument(Document):
    """Represents a Reddit document."""

    def __init__(self, titre="", auteur="", date="", url="", texte="", num_comments=0):
        """Initializes a RedditDocument instance.

        Args:
            titre (str, optional): The title of the document.
            auteur (str, optional): The author of the document.
            date (str, optional): The publication date of the document.
            url (str, optional): The URL of the document.
            texte (str, optional): The text content of the document.
            num_comments (int, optional): The number of comments on the document.
        """
        super().__init__(titre, auteur, date, url, texte)  # Call the parent constructor
        self.num_comments = num_comments  # Store the number of comments
        self.type = "Reddit"  # Specific type for Reddit documents

    def get_num_comments(self):
        """Gets the number of comments on the document.

        Returns:
            int: The number of comments.
        """
        return self.num_comments

    def __str__(self):
        """Returns a string representation of the Reddit document.

        Returns:
            str: A string representation including title and number of comments.
        """
        return f"{super().__str__()}, Comments: {self.num_comments}"


class ArxivDocument(Document):
    """Represents an Arxiv document."""

    def __init__(self, titre="", auteurs=[""], date="", url="", texte=""):
        """Initializes an ArxivDocument instance.

        Args:
            titre (str, optional): The title of the document.
            auteurs (list, optional): A list of authors for the document.
            date (str, optional): The publication date of the document.
            url (str, optional): The URL of the document.
            texte (str, optional): The text content of the document.
        """
        super().__init__(
            titre, auteurs[0], date, url, texte
        )  # Call the parent constructor
        self.authors = auteurs  # Store the list of co-authors
        self.type = "Arxiv"  # Specific type for Arxiv documents

    def get_authors(self):
        """Gets the list of authors for the document.

        Returns:
            list: A list of authors.
        """
        return self.authors

    def __str__(self):
        """Returns a string representation of the Arxiv document.

        Returns:
            str: A string representation including title and authors.
        """
        return f"{super().__str__()}, Authors: {', '.join(self.authors)}"
