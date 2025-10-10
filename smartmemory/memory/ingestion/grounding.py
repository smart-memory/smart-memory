"""
Helpers for grounding notes to external sources (Wikipedia, others in future).
"""
from smartmemory.integration.wikipedia_client import WikipediaClient


def ground_note(note: str, language: str = "en") -> dict:
    """
    Attempt to ground a note using external sources (Wikipedia, others in future).

    Args:
        note (str): The note content to ground.
        language (str): Language code for Wikipedia (default 'en').
    Returns:
        dict: Dict with keys relevant to the source (e.g., 'wikipedia_url', 'wikipedia_snippet', 'wikipedia_summary').
    """
    client = WikipediaClient(language=language)
    results = client.search(note, limit=1)
    if results:
        title = results[0]['title']
        url = f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}"
        snippet = results[0]['snippet']
        summary = client.get_summary(title)
        return {
            "wikipedia_url": url,
            "wikipedia_snippet": snippet,
            "wikipedia_title": title,
            "wikipedia_summary": summary
        }
    return {}
