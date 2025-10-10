"""
Wikipedia tool functions for agentic use. Wraps WikipediaClient methods as serializable tools for LLM/agent workflows.
"""
from smartmemory.integration.wikipedia_client import WikipediaClient


def wikipedia_search(query: str, limit: int = 5, language: str = "en") -> list:
    """
    Search Wikipedia for articles matching a query string.

    Args:
        query (str): Search query string.
        limit (int): Maximum number of results to return.
        language (str): Wikipedia language code (default: 'en').
    Returns:
        list: List of matching article summaries.
    """
    client = WikipediaClient(language=language)
    return client.search(query, limit=limit)


from typing import Union, List, Dict


def wikipedia_summary(title: Union[str, List[str]], language: str = "en") -> Dict[str, str]:
    """
    Get a summary of one or more Wikipedia articles by title.

    Args:
        title (str or list of str): Article title or list of titles. If a string is provided, it will be treated as a single-item list.
        language (str): Wikipedia language code (default: 'en').
    Returns:
        dict: Mapping from each title to its summary string. Always returns a dict, even for a single input.
    
    Note:
        This function is LLM/agent-tool friendly: always returns a dict mapping title(s) to summary string(s).
    """
    client = WikipediaClient(language=language)
    if isinstance(title, str):
        title = [title]
    return {t: client.get_summary(t) for t in title}


def wikipedia_article(title: Union[str, List[str]], language: str = "en") -> Dict[str, dict]:
    """
    Get the full content and metadata of one or more Wikipedia articles.

    Args:
        title (str or list of str): Article title or list of titles. If a string is provided, it will be treated as a single-item list.
        language (str): Wikipedia language code (default: 'en').
    Returns:
        dict: Mapping from each title to its article dict (summary, categories, url, etc.). Always returns a dict, even for a single input.
    
    Note:
        This function is LLM/agent-tool friendly: always returns a dict mapping title(s) to article dict(s).
    """
    client = WikipediaClient(language=language)
    if isinstance(title, str):
        title = [title]
    return {t: client.get_article(t) for t in title}


def wikipedia_sections(title: str, language: str = "en") -> list:
    """
    Get the section titles of a Wikipedia article.

    Args:
        title (str): The title of the Wikipedia article.
        language (str): Wikipedia language code (default: 'en').
    Returns:
        list: List of section titles.
    """
    client = WikipediaClient(language=language)
    return client.get_sections(title)


def wikipedia_links(title: str, language: str = "en") -> list:
    """
    Get all links (internal and external) in a Wikipedia article.

    Args:
        title (str): The title of the Wikipedia article.
        language (str): Wikipedia language code (default: 'en').
    Returns:
        list: List of links (URLs or article titles).
    """
    client = WikipediaClient(language=language)
    return client.get_links(title)


def wikipedia_related_topics(title: str, limit: int = 10, language: str = "en") -> list:
    """
    Get topics related to a Wikipedia article based on links and categories.

    Args:
        title (str): The title of the Wikipedia article.
        limit (int): Maximum number of related topics to return.
        language (str): Wikipedia language code (default: 'en').
    Returns:
        list: List of related topic titles.
    """
    client = WikipediaClient(language=language)
    return client.get_related_topics(title, limit=limit)
