import argparse
import os
import string
import pickle
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


    print("cat" in "concatinate")

    print(string.punctuation)
    print(str.maketrans("ab", "xy", "?!"))

    index = {"alien": {1, 4, 7}, "space": {1, 9}} # a dict of sets

    with open("index.pkl", "wb") as f: # "wb" = write binary
        pickle.dump(index, f)

    with open("index.pkl", "rb") as f: # "rb" = read binary
        loaded = pickle.load(f)

    print(loaded == index)      # True: same dict, same sets 



if __name__ == "__main__":
    main()
