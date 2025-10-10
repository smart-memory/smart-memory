"""
Example: Creating a Custom Enricher Plugin

This example demonstrates how to create a custom enricher plugin that
adds sentiment analysis and keyword extraction to memory items.
"""

from typing import Optional, Dict, Any
from smartmemory.plugins.base import EnricherPlugin, PluginMetadata


class CustomSentimentEnricher(EnricherPlugin):
    """
    Custom enricher that adds detailed sentiment analysis to memory items.
    
    This enricher analyzes the emotional tone of text and extracts
    key sentiment indicators.
    """
    
    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery and registration."""
        return PluginMetadata(
            name="custom_sentiment_enricher",
            version="1.0.0",
            author="Your Name",
            description="Advanced sentiment analysis enricher",
            plugin_type="enricher",
            dependencies=["textblob>=0.15.0"],  # Optional dependencies
            min_smartmemory_version="0.1.0",
            tags=["sentiment", "nlp", "emotion"]
        )
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enricher with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.threshold = config.get('threshold', 0.5) if config else 0.5
        self._analyzer = None
    
    def _load_analyzer(self):
        """Lazy load the sentiment analyzer."""
        if self._analyzer is None:
            try:
                from textblob import TextBlob
                self._analyzer = TextBlob
            except ImportError:
                # Fallback to simple sentiment analysis
                self._analyzer = None
    
    def enrich(self, item, node_ids: Optional[list] = None) -> Dict[str, Any]:
        """
        Enrich a memory item with sentiment analysis.
        
        Args:
            item: The memory item to enrich
            node_ids: Optional list of related node IDs
        
        Returns:
            Dictionary with enrichment data
        """
        self._load_analyzer()
        
        # Get text content
        text = getattr(item, 'content', str(item))
        
        if not text:
            return {}
        
        # Perform sentiment analysis
        if self._analyzer:
            blob = self._analyzer(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Classify sentiment
            if polarity > self.threshold:
                sentiment = "positive"
            elif polarity < -self.threshold:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            return {
                'sentiment': sentiment,
                'sentiment_score': polarity,
                'subjectivity': subjectivity,
                'is_subjective': subjectivity > 0.5,
                'enricher': 'custom_sentiment_enricher'
            }
        else:
            # Simple fallback sentiment
            positive_words = ['good', 'great', 'excellent', 'happy', 'love']
            negative_words = ['bad', 'terrible', 'awful', 'sad', 'hate']
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                sentiment = "positive"
            elif neg_count > pos_count:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            return {
                'sentiment': sentiment,
                'enricher': 'custom_sentiment_enricher',
                'method': 'fallback'
            }


class KeywordEnricher(EnricherPlugin):
    """
    Simple enricher that extracts keywords from text.
    """
    
    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="keyword_enricher",
            version="1.0.0",
            author="Your Name",
            description="Extracts keywords from text",
            plugin_type="enricher"
        )
    
    def enrich(self, item, node_ids: Optional[list] = None) -> Dict[str, Any]:
        """Extract keywords from the item content."""
        text = getattr(item, 'content', str(item))
        
        # Simple keyword extraction (you could use more sophisticated methods)
        words = text.lower().split()
        
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Get unique keywords
        unique_keywords = list(set(keywords))[:10]  # Top 10 unique keywords
        
        return {
            'keywords': unique_keywords,
            'keyword_count': len(unique_keywords),
            'enricher': 'keyword_enricher'
        }


# Example usage
if __name__ == "__main__":
    from smartmemory.models.memory_item import MemoryItem
    
    # Create a memory item
    item = MemoryItem(
        content="I absolutely love this new feature! It's fantastic and makes everything so much easier.",
        memory_type="episodic"
    )
    
    # Create and use the custom enricher
    enricher = CustomSentimentEnricher(config={'threshold': 0.3})
    result = enricher.enrich(item)
    
    print("Enrichment Result:")
    print(f"  Sentiment: {result.get('sentiment')}")
    print(f"  Score: {result.get('sentiment_score', 'N/A')}")
    print(f"  Subjectivity: {result.get('subjectivity', 'N/A')}")
    print()
    
    # Use keyword enricher
    keyword_enricher = KeywordEnricher()
    keywords = keyword_enricher.enrich(item)
    
    print("Keywords:")
    print(f"  {keywords.get('keywords')}")
    print(f"  Count: {keywords.get('keyword_count')}")
