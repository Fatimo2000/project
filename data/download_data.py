import os  # Standard library for interacting with the operating system
import dotenv  # Library for loading environment variables from a .env file
import datetime  # Library for handling date and time
from tqdm import tqdm  # Library for displaying progress bars

# Correction de G. Poux-Médard, 2021-2022

# =============== PARTIE 1 =============
# =============== 1.1 : REDDIT ===============
import praw  # Library for accessing the Reddit API

# =============== 1.2 : ArXiv ===============
import urllib  # Library for URL handling
import xmltodict  # Library for parsing XML data
import pandas as pd  # Library for data manipulation and analysis

# Load environment variables from .env file
dotenv.load_dotenv()


def get_data_from_reddit(max_results):
    """Fetch data from Reddit.

    Args:
        max_results (int): The maximum number of posts to retrieve.

    Returns:
        list: A list of dictionaries containing post data.
    """
    # Load Environment Variables for Reddit API credentials
    dotenv.load_dotenv()

    # Fetch Reddit API credentials from environment variables
    reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
    reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    reddit_user_agent = os.getenv("REDDIT_USER_AGENT")

    # Initialize Reddit instance with provided credentials
    reddit = praw.Reddit(
        client_id=reddit_client_id,
        client_secret=reddit_client_secret,
        user_agent=reddit_user_agent,
    )

    # Fetch hot posts from the "Coronavirus" subreddit with a limit on results
    hot_posts = reddit.subreddit("Coronavirus").hot(limit=max_results)

    # List to store raw document entries
    docs_bruts = []
    for post in tqdm(
        iterable=hot_posts,
        total=max_results,
        unit="post",
        desc="Downloading posts from reddit ...",
    ):
        # Extracting relevant information from each post
        titre = post.title.replace("\n", "")  # Clean title by removing newlines
        auteur = str(post.author)  # Author of the post
        date = datetime.datetime.fromtimestamp(post.created).strftime(
            "%Y/%m/%d"
        )  # Formatting the creation date
        url = "https://www.reddit.com/" + post.permalink  # Constructing the post URL
        texte = post.selftext.replace(
            "\n", " "
        )  # Clean text by replacing newlines with spaces
        num_comments = len(
            post.comments.list()
        )  # Count the number of comments on the post

        # Create a dictionary to store the post data
        entry = {
            "titre": titre,
            "authors": auteur,
            "date": date,
            "texte": texte,
            "num_comments": num_comments,
            "url": url,
            "source": "Reddit",
        }
        docs_bruts.append(entry)  # Append the entry to the list

    return docs_bruts  # Return the list of posts


def get_data_from_arxiv(max_results):
    """Fetch data from ArXiv.

    Args:
        max_results (int): The maximum number of entries to retrieve.

    Returns:
        list: A list of dictionaries containing entry data.
    """
    # Define search terms for querying ArXiv
    query_terms = ["Coronavirus"]

    # Construct the ArXiv API query URL with the search terms and result limits
    url = f'http://export.arxiv.org/api/query?search_query=all:{"+".join(query_terms)}&start=0&max_results={max_results}'
    data = urllib.request.urlopen(url)  # Open the URL and fetch data

    # Parse the XML response into a dictionary format
    data = xmltodict.parse(data.read().decode("utf-8"))

    # List to store raw document entries
    docs_bruts = []
    for entry in tqdm(
        iterable=data["feed"]["entry"],
        total=max_results,
        unit="post",
        desc="Downloading posts from arxiv ...",
    ):
        # Extracting relevant information from each entry
        titre = entry["title"].replace("\n", "")  # Clean title by removing newlines
        date = datetime.datetime.strptime(
            entry["published"], "%Y-%m-%dT%H:%M:%SZ"
        ).strftime("%Y/%m/%d")  # Formatting the publication date
        summary = entry["summary"].replace(
            "\n", " "
        )  # Clean summary text by replacing newlines with spaces

        # Handling the author information, which can be a single dict or a list of dicts
        if isinstance(entry["author"], dict):
            authors = entry["author"]["name"]  # Single author
        elif isinstance(entry["author"], list):
            authors = [a["name"] for a in entry["author"]]  # List of authors
            authors = ",".join(authors)  # Join authors into a single string

        url = entry["id"]  # Get the URL of the entry
        entry = {
            "titre": titre,
            "authors": authors,
            "date": date,
            "url": url,
            "summary": summary,
            "source": "ArXiv",
        }

        docs_bruts.append(entry)  # Append the entry to the list
    return docs_bruts  # Return the list of entries


def download_data(max_reddit_results=1000, max_arxiv_results=1000):
    """Download data from Reddit and ArXiv.

    Args:
        max_reddit_results (int): Maximum number of Reddit posts to fetch.
        max_arxiv_results (int): Maximum number of ArXiv entries to fetch.
    """
    # Fetch data from Reddit and ArXiv
    df_data = [
        *get_data_from_reddit(max_results=max_reddit_results),  # Fetch Reddit data
        *get_data_from_arxiv(max_results=max_arxiv_results),  # Fetch ArXiv data
    ]

    # Create a DataFrame from the combined data
    df = pd.DataFrame(df_data)
    df["id"] = range(1, len(df) + 1)  # Add an ID column to the DataFrame

    # Save the DataFrame to a CSV file
    with open("./data/data.csv", mode="wb") as file:
        df.to_csv(file, sep="\t", index=False)  # Save as tab-separated values


def load_data():
    """Load data from the CSV file.

    Returns:
        DataFrame: A pandas DataFrame containing the loaded data.
    """
    if not os.path.exists("./data/data.csv"):
        download_data()  # If the file doesn't exist, download data

    data = None
    with open("./data/data.csv", mode="rb") as file:
        data = pd.read_csv(file, sep="\t")  # Read the data from the CSV file
    return data  # Return the loaded data


if __name__ == "__main__":
    download_data()  # Execute the download_data function if the script is run directly
