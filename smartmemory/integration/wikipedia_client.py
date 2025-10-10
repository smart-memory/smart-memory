"""
Wikipedia API client and utilities for search and summary.
"""
import requests
import wikipediaapi


class WikipediaClient:
    def __init__(self, language: str = "en"):
        self.language = language
        self.user_agent = "AgenticMemoryGrounder/0.1"
        self.wiki = wikipediaapi.Wikipedia(
            user_agent=self.user_agent,
            language=language,
            extract_format=wikipediaapi.ExtractFormat.WIKI
        )
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"

    def search(self, query: str, limit: int = 1):
        """Search Wikipedia for articles matching a query."""
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'utf8': 1,
            'srsearch': query,
            'srlimit': limit
        }
        try:
            response = requests.get(self.api_url, params=params, timeout=4)
            response.raise_for_status()
            data = response.json()
            results = data.get('query') or {}.get('search', [])
            return results
        except Exception:
            return []

    def get_summary(self, title: str) -> str:
        """Get a summary of a Wikipedia article by title."""
        page = self.wiki.page(title)
        if page.exists():
            return page.summary
        return ""

    def get_article(self, title: str) -> dict:
        """Get the full content and metadata of a Wikipedia article."""
        try:
            page = self.wiki.page(title)
            if not page.exists():
                return {
                    'title': title,
                    'exists': False,
                    'error': 'Page does not exist'
                }
            sections = self._extract_sections(page.sections)
            categories = [cat for cat in page.categories.keys()]
            links = [link for link in page.links.keys()]
            return {
                'title': page.title,
                'pageid': page.pageid,
                'summary': page.summary,
                'text': page.text,
                'url': page.fullurl,
                'sections': sections,
                'categories': categories,
                'links': links[:100],
                'exists': True
            }
        except Exception as e:
            return {
                'title': title,
                'exists': False,
                'error': str(e)
            }

    def get_sections(self, title: str) -> list:
        """Get the sections of a Wikipedia article."""
        try:
            page = self.wiki.page(title)
            if not page.exists():
                return []
            return self._extract_sections(page.sections)
        except Exception:
            return []

    def get_links(self, title: str) -> list:
        """Get the links in a Wikipedia article."""
        try:
            page = self.wiki.page(title)
            if not page.exists():
                return []
            return [link for link in page.links.keys()]
        except Exception:
            return []

    def get_related_topics(self, title: str, limit: int = 10) -> list:
        """Get topics related to a Wikipedia article based on links and categories."""
        try:
            page = self.wiki.page(title)
            if not page.exists():
                return []
            links = list(page.links.keys())
            categories = list(page.categories.keys())
            related = []
            for link in links[:limit]:
                link_page = self.wiki.page(link)
                if link_page.exists():
                    related.append({
                        'title': link,
                        'summary': link_page.summary[:200] + '...' if len(link_page.summary) > 200 else link_page.summary,
                        'url': link_page.fullurl,
                        'type': 'link'
                    })
                if len(related) >= limit:
                    break
            remaining = limit - len(related)
            if remaining > 0:
                for category in categories[:remaining]:
                    clean_category = category.replace("Category:", "")
                    related.append({
                        'title': clean_category,
                        'type': 'category'
                    })
            return related
        except Exception:
            return []

    def _extract_sections(self, sections, level=0) -> list:
        """Extract sections recursively from a Wikipedia article."""
        result = []
        for section in sections:
            section_data = {
                'title': section.title,
                'level': level,
                'text': section.text,
                'sections': self._extract_sections(section.sections, level + 1)
            }
            result.append(section_data)
        return result
