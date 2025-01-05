from Classes import Document


class DocumentFactory:
    @staticmethod
    def create_arxiv_document(titre="", authors=[""], date="", url="", summary=""):
        return ArxivDocument(titre, authors, date, url, summary)

    @staticmethod
    def create_reddit_document(
        titre="", auteur="", date="", url="", texte="", num_comments=0
    ):
        return RedditDocument(
            titre, auteur, date, url, texte, num_comments=num_comments
        )


class RedditDocument(Document):
    def __init__(self, titre="", auteur="", date="", url="", texte="", num_comments=0):
        super().__init__(titre, auteur, date, url, texte)  # Call the parent constructor
        self.num_comments = num_comments
        self.type = "Reddit"  # Specific type for Reddit documents

    def get_num_comments(self):
        return self.num_comments

    def __str__(self):
        return f"{super().__str__()}, Comments: {self.num_comments}"


class ArxivDocument(Document):
    def __init__(self, titre="", auteurs=[""], date="", url="", texte=""):
        super().__init__(
            titre, auteurs[0], date, url, texte
        )  # Call the parent constructor
        self.authors = auteurs  # List of co-authors
        self.type = "Arxiv"  # Specific type for Arxiv documents

    def get_authors(self):
        return self.authors

    def __str__(self):
        return f"{super().__str__()}, Authors: {', '.join(self.authors)}"
