import argparse
import os
from nltk.stem import PorterStemmer
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")


STEMMER = PorterStemmer()

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    arg = parser.parse_args()



    print(arg.command)
    print(arg.query)
    print(__file__)
    print(os.path.dirname(__file__))
    print(PROJECT_ROOT)
    print(DATA_PATH)



if __name__ == "__main__":
    main()
