import datetime
# Correction de G. Poux-Médard, 2021-2022


# =============== 2.1 : La classe Document ===============
class Document:
    # Initialisation des variables de la classe
    def __init__(
        self, titre: str, auteur: str, date: datetime.date, url: str, texte: str
    ):
        self.titre = titre
        self.auteur = auteur
        self.date = date
        self.url = url
        self.texte = texte
        self.type = "Document"

    # =============== 2.2 : REPRESENTATIONS ===============
    # Fonction qui renvoie le texte à afficher lorsqu'on tape repr(classe)
    def __repr__(self):
        return f"Titre: {self.titre}\nAuteur: {self.auteur}\nDate: {self.date}\nURL: {self.url}\nTexte: {self.texte}\nType: {self.type}\n"

    # Fonction qui renvoie le texte à afficher lorsqu'on tape str(classe)
    def __str__(self):
        return f"{self.titre}, par {self.auteur}"

    def getType(self):
        return self.type


# =============== 2.4 : La classe Author ===============
class Author:
    def __init__(self, name: str):
        self.name = name
        self.ndoc = 0
        self.production = []

    # =============== 2.5 : ADD ===============
    def add(self, production):
        self.ndoc += 1
        self.production.append(production)

    def __str__(self):
        return f"Auteur : {self.name}\t# productions : {self.ndoc}"

    def __repr__(self):
        return self.name
