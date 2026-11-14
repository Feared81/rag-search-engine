import os
import pickle
import string
import math

from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    STOPWORDS_PATH,
    Movie,
    load_movies,
)


class InvertedIndex:
    def __init__(self):
            self.index: dict[str, set[int]] = defaultdict(set)
            self.docmap: dict[int, Movie] = {}
            self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
            
            self.index_path = os.path.join(CACHE_DIR, "index.pkl")
            self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
            self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies[doc_id][term]

    def get_idf(self, term: str) -> float:
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[term])
        return math.log((doc_count + 1)/(term_doc_count + 1))

    def get_tfidf(self, doc_id: int, term: str) -> float:
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.index[term])
        return math.log((N - df + 0.5)/(df + 0.5) + 1)
        
    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(doc_ids)
        
    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.__add_document(doc_id, doc_description)
            self.docmap[doc_id] = m

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        try:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
            with open(self.tf_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Index not found. Run the 'build' command first."
            ) from None



def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Movie]:
    idx = InvertedIndex()
    idx.load()

    query_tokens = tokenize_text(query)
    
    seen: set[int] = set()
    results: list[Movie] = []
    for token in query_tokens:
        matching_doc_ids = idx.get_documents(token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results
    return results

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return [preprocess_text(word) for word in f.read().splitlines()]

STOPWORDS = load_stopwords()

def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    filtered_words = []
    for word in valid_tokens:
        if word not in STOPWORDS:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words

def tokenize_single_term(term: str) -> str:
    token = tokenize_text(term)
    if len(token) != 1:
        raise ValueError("term must be a single token")
    return token[0]

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()

def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, tokenize_single_term(term))

def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(tokenize_single_term(term))

def tfidf_command(doc_id: int, term: str):
    idx = InvertedIndex()
    idx.load()
    return idx.get_tfidf(doc_id, tokenize_single_term(term))

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(tokenize_single_term(term))
