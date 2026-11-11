import string

from .search_utils import DEFAULT_SEARCH_LIMIT, STOPWORDS_PATH, Movie, load_movies


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Movie]:
    movies = load_movies()
    results = []
    for movie in movies:
        query_tokens = tokenize_text(query)
        title_tokens = tokenize_text(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break

    return results

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    pass

def tokenize_text(text: str) -> list[str]:
    pass