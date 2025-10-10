"""
Web operations tools for Maya assistant.
Provides HTTP requests, web scraping, and API interaction capabilities.
"""
import json
import requests
from typing import Optional, Dict, Any
from urllib.parse import urlparse


def http_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> str:
    """
    Make an HTTP GET request.
    
    Args:
        url (str): URL to request
        headers (dict, optional): HTTP headers
        timeout (int): Request timeout in seconds (default: 30)
        
    Returns:
        str: Response content or error message
    """
    try:
        # Basic URL validation
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return f"Error: Invalid URL: {url}"

        # Security check - prevent internal network access
        if parsed.hostname in ['localhost', '127.0.0.1'] or parsed.hostname.startswith('192.168.'):
            return f"Error: Access to internal networks not allowed: {url}"

        response = requests.get(url, headers=headers or {}, timeout=timeout)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()

        if 'application/json' in content_type:
            try:
                json_data = response.json()
                return f"HTTP GET {url} (Status: {response.status_code}):\n{json.dumps(json_data, indent=2)}"
            except:
                pass

        return f"HTTP GET {url} (Status: {response.status_code}):\n{response.text[:2000]}{'...' if len(response.text) > 2000 else ''}"

    except requests.exceptions.Timeout:
        return f"Error: Request timeout for {url}"
    except requests.exceptions.ConnectionError:
        return f"Error: Connection failed for {url}"
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {e.response.status_code} for {url}: {e.response.text[:500]}"
    except Exception as e:
        return f"Error making request to {url}: {str(e)}"


http_get.tags = ["web", "http", "get", "api"]
http_get.args_schema = {"url": str, "headers": dict, "timeout": int}


def http_post(url: str, data: Optional[Dict[str, Any]] = None, json_data: Optional[Dict[str, Any]] = None,
              headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> str:
    """
    Make an HTTP POST request.
    
    Args:
        url (str): URL to request
        data (dict, optional): Form data to send
        json_data (dict, optional): JSON data to send
        headers (dict, optional): HTTP headers
        timeout (int): Request timeout in seconds (default: 30)
        
    Returns:
        str: Response content or error message
    """
    try:
        # Basic URL validation
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return f"Error: Invalid URL: {url}"

        # Security check - prevent internal network access
        if parsed.hostname in ['localhost', '127.0.0.1'] or parsed.hostname.startswith('192.168.'):
            return f"Error: Access to internal networks not allowed: {url}"

        kwargs = {'timeout': timeout, 'headers': headers or {}}

        if json_data:
            kwargs['json'] = json_data
        elif data:
            kwargs['data'] = data

        response = requests.post(url, **kwargs)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()

        if 'application/json' in content_type:
            try:
                json_response = response.json()
                return f"HTTP POST {url} (Status: {response.status_code}):\n{json.dumps(json_response, indent=2)}"
            except:
                pass

        return f"HTTP POST {url} (Status: {response.status_code}):\n{response.text[:2000]}{'...' if len(response.text) > 2000 else ''}"

    except requests.exceptions.Timeout:
        return f"Error: Request timeout for {url}"
    except requests.exceptions.ConnectionError:
        return f"Error: Connection failed for {url}"
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {e.response.status_code} for {url}: {e.response.text[:500]}"
    except Exception as e:
        return f"Error making POST request to {url}: {str(e)}"


http_post.tags = ["web", "http", "post", "api"]
http_post.args_schema = {"url": str, "data": dict, "json_data": dict, "headers": dict, "timeout": int}


def fetch_webpage_text(url: str, max_length: int = 5000) -> str:
    """
    Fetch and extract text content from a webpage.
    
    Args:
        url (str): URL of the webpage
        max_length (int): Maximum length of text to return (default: 5000)
        
    Returns:
        str: Extracted text content or error message
    """
    try:
        # Basic URL validation
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return f"Error: Invalid URL: {url}"

        # Security check
        if parsed.hostname in ['localhost', '127.0.0.1'] or parsed.hostname.startswith('192.168.'):
            return f"Error: Access to internal networks not allowed: {url}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; Maya Assistant; +https://example.com/bot)'
        }

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # Try to extract text content
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

        except ImportError:
            # Fallback without BeautifulSoup
            text = response.text

        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length] + "..."

        return f"Text content from {url}:\n{text}"

    except requests.exceptions.Timeout:
        return f"Error: Request timeout for {url}"
    except requests.exceptions.ConnectionError:
        return f"Error: Connection failed for {url}"
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {e.response.status_code} for {url}"
    except Exception as e:
        return f"Error fetching webpage {url}: {str(e)}"


fetch_webpage_text.tags = ["web", "scraping", "text", "html"]
fetch_webpage_text.args_schema = {"url": str, "max_length": int}


def search_web(query: str, num_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo (no API key required).
    
    Args:
        query (str): Search query
        num_results (int): Number of results to return (default: 5)
        
    Returns:
        str: Search results or error message
    """
    try:
        # Use DuckDuckGo instant answer API
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1'
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        results = []

        # Add instant answer if available
        if data.get('Abstract'):
            results.append(f"Summary: {data['Abstract']}")
            if data.get('AbstractURL'):
                results.append(f"Source: {data['AbstractURL']}")

        # Add related topics
        if data.get('RelatedTopics'):
            results.append("\nRelated topics:")
            for i, topic in enumerate(data['RelatedTopics'][:num_results]):
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append(f"{i + 1}. {topic['Text']}")
                    if topic.get('FirstURL'):
                        results.append(f"   URL: {topic['FirstURL']}")

        if not results:
            return f"No results found for query: {query}"

        return f"Web search results for '{query}':\n" + "\n".join(results)

    except Exception as e:
        return f"Error searching web for '{query}': {str(e)}"


search_web.tags = ["web", "search", "duckduckgo"]
search_web.args_schema = {"query": str, "num_results": int}


def download_file(url: str, filename: Optional[str] = None, max_size: int = 10 * 1024 * 1024) -> str:
    """
    Download a file from a URL.
    
    Args:
        url (str): URL of the file to download
        filename (str, optional): Local filename to save as
        max_size (int): Maximum file size in bytes (default: 10MB)
        
    Returns:
        str: Success message with file info or error message
    """
    try:
        # Basic URL validation
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return f"Error: Invalid URL: {url}"

        # Security check
        if parsed.hostname in ['localhost', '127.0.0.1'] or parsed.hostname.startswith('192.168.'):
            return f"Error: Access to internal networks not allowed: {url}"

        # Get filename from URL if not provided
        if not filename:
            filename = url.split('/')[-1] or 'downloaded_file'

        # Security check for filename
        if '/' in filename or '\\' in filename or filename.startswith('.'):
            return f"Error: Invalid filename: {filename}"

        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Check content length
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > max_size:
            return f"Error: File too large ({content_length} bytes, max {max_size})"

        # Download with size limit
        downloaded = 0
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    downloaded += len(chunk)
                    if downloaded > max_size:
                        f.close()
                        import os
                        os.remove(filename)
                        return f"Error: File too large (exceeded {max_size} bytes)"
                    f.write(chunk)

        return f"Successfully downloaded {url} as {filename} ({downloaded} bytes)"

    except requests.exceptions.Timeout:
        return f"Error: Request timeout for {url}"
    except requests.exceptions.ConnectionError:
        return f"Error: Connection failed for {url}"
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {e.response.status_code} for {url}"
    except Exception as e:
        return f"Error downloading file from {url}: {str(e)}"


download_file.tags = ["web", "download", "file"]
download_file.args_schema = {"url": str, "filename": str, "max_size": int}
