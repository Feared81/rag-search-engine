import argparse
import json
import string
from nltk.stem import PorterStemmer

TRANSLATION_TABLE = str.maketrans("","", string.punctuation)
STEMMER = PorterStemmer()

def preprocess_text(text: str) -> str:
    return text.lower().translate(TRANSLATION_TABLE)

def load_movies(path: str = "data/movies.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)["movies"]

def load_stopwords(path: str = "data/stopwords.txt") -> set[str]:
    with open(path) as f:
        return {preprocess_text(word) for word in f.read().splitlines()}

def tokenize(text: str, stopwords: set[str]) -> list[str]:
    tokens = preprocess_text(text).split()
    tokens = [t for t in tokens if t not in stopwords]
    return [STEMMER.stem(t) for t in tokens]


def token_matching(query: str, title: str, stopwords: set[str]) -> bool:
    query_tokens = tokenize(query, stopwords)
    title_tokens = tokenize(title, stopwords)
    return any(q in t for q in query_tokens for t in title_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            movies = load_movies()
            stopwords = load_stopwords()

            results = [
                movie["title"]
                for movie in movies
                if token_matching(args.query, movie["title"], stopwords)
            ]

            print(f"Searching for: {args.query}")
            for i, title in enumerate(results[:5], start=1):
                print(f"{i}. {title}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
