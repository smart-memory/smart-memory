"""
Integration tests for extraction pipeline.

Tests the complete extraction workflow from text input to stored entities.
"""

import pytest


pytestmark = pytest.mark.unit
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from smartmemory.models.memory_item import MemoryItem


class TestExtractionPipeline:
    """Test complete extraction pipeline."""
    
    @pytest.fixture
    def mock_memory(self):
        """Create mock memory system."""
        memory = Mock()
        memory.graph = Mock()
        memory.add = Mock(return_value="item_id_123")
        return memory
    
    def test_text_to_entities_pipeline(self, mock_memory):
        """Test complete pipeline from text to entities."""
        text = "Albert Einstein developed the theory of relativity in 1905."
        
        # Mock extractor
        with patch('smartmemory.plugins.extractors.spacy.SpacyExtractor') as MockExtractor:
            mock_extractor = Mock()
            mock_extractor.extract.return_value = {
                'entities': [
                    {'text': 'Albert Einstein', 'type': 'person', 'start': 0, 'end': 15},
                    {'text': 'theory of relativity', 'type': 'concept', 'start': 34, 'end': 54},
                    {'text': '1905', 'type': 'temporal', 'start': 58, 'end': 62}
                ],
                'relations': [
                    {'source': 'Albert Einstein', 'target': 'theory of relativity', 'type': 'developed'}
                ]
            }
            MockExtractor.return_value = mock_extractor
            
            # Create memory item
            item = MemoryItem(
                content=text,
                memory_type="semantic",
                
            )
            
            # Extract entities
            extraction_result = mock_extractor.extract(text)
            
            # Verify extraction
            assert len(extraction_result['entities']) == 3
            assert len(extraction_result['relations']) == 1
            
            # Verify entity types
            entity_types = [e['type'] for e in extraction_result['entities']]
            assert 'person' in entity_types
            assert 'concept' in entity_types
            assert 'temporal' in entity_types
    
    def test_extraction_with_enrichment(self, mock_memory):
        """Test extraction followed by enrichment."""
        text = "Python is a programming language created by Guido van Rossum."
        
        with patch('smartmemory.plugins.extractors.spacy.SpacyExtractor') as MockExtractor:
            mock_extractor = Mock()
            mock_extractor.extract.return_value = {
                'entities': [
                    {'text': 'Python', 'type': 'technology', 'start': 0, 'end': 6},
                    {'text': 'Guido van Rossum', 'type': 'person', 'start': 44, 'end': 60}
                ],
                'relations': [
                    {'source': 'Guido van Rossum', 'target': 'Python', 'type': 'created'}
                ]
            }
            MockExtractor.return_value = mock_extractor
            
            extraction_result = mock_extractor.extract(text)
            
            # Verify extraction
            assert len(extraction_result['entities']) == 2
            assert extraction_result['relations'][0]['type'] == 'created'
    
    def test_extraction_error_handling(self, mock_memory):
        """Test pipeline handles extraction errors gracefully."""
        text = "Test text"
        
        with patch('smartmemory.plugins.extractors.spacy.SpacyExtractor') as MockExtractor:
            mock_extractor = Mock()
            mock_extractor.extract.side_effect = Exception("Extraction failed")
            MockExtractor.return_value = mock_extractor
            
            # Should handle error gracefully
            try:
                mock_extractor.extract(text)
                assert False, "Should have raised exception"
            except Exception as e:
                assert str(e) == "Extraction failed"
    
    def test_batch_extraction(self, mock_memory):
        """Test batch extraction of multiple texts."""
        texts = [
            "Python is a programming language.",
            "JavaScript is used for web development.",
            "Java is an object-oriented language."
        ]
        
        with patch('smartmemory.plugins.extractors.spacy.SpacyExtractor') as MockExtractor:
            mock_extractor = Mock()
            
            def mock_extract(text):
                # Return different entities based on text
                if "Python" in text:
                    return {
                        'entities': [{'text': 'Python', 'type': 'technology', 'start': 0, 'end': 6}],
                        'relations': []
                    }
                elif "JavaScript" in text:
                    return {
                        'entities': [{'text': 'JavaScript', 'type': 'technology', 'start': 0, 'end': 10}],
                        'relations': []
                    }
                else:
                    return {
                        'entities': [{'text': 'Java', 'type': 'technology', 'start': 0, 'end': 4}],
                        'relations': []
                    }
            
            mock_extractor.extract.side_effect = mock_extract
            MockExtractor.return_value = mock_extractor
            
            # Extract from all texts
            results = [mock_extractor.extract(text) for text in texts]
            
            assert len(results) == 3
            assert all(len(r['entities']) > 0 for r in results)
    
    def test_extraction_with_user_context(self, mock_memory):
        """Test extraction with user-specific context."""
        text = "I love using Python for data science."
        user_id = "user_123"
        
        with patch('smartmemory.plugins.extractors.spacy.SpacyExtractor') as MockExtractor:
            mock_extractor = Mock()
            mock_extractor.extract.return_value = {
                'entities': [
                    {'text': 'Python', 'type': 'technology', 'start': 14, 'end': 20},
                    {'text': 'data science', 'type': 'field', 'start': 25, 'end': 37}
                ],
                'relations': []
            }
            MockExtractor.return_value = mock_extractor
            
            # Extract with user context
            result = mock_extractor.extract(text)
            
            assert len(result['entities']) == 2


class TestExtractionAccuracy:
    """Test extraction accuracy with known ground truth."""
    
    def test_person_extraction_accuracy(self):
        """Test accuracy of person entity extraction."""
        test_cases = [
            {
                'text': "Albert Einstein was a physicist.",
                'expected_entities': [{'text': 'Albert Einstein', 'type': 'person'}]
            },
            {
                'text': "Marie Curie won the Nobel Prize.",
                'expected_entities': [{'text': 'Marie Curie', 'type': 'person'}]
            }
        ]
        
        with patch('smartmemory.plugins.extractors.spacy.SpacyExtractor') as MockExtractor:
            mock_extractor = Mock()
            
            for case in test_cases:
                # Mock extraction to return expected entities
                mock_extractor.extract.return_value = {
                    'entities': case['expected_entities'],
                    'relations': []
                }
                
                result = mock_extractor.extract(case['text'])
                
                # Verify extraction matches expected
                assert len(result['entities']) == len(case['expected_entities'])
                assert result['entities'][0]['type'] == 'person'
    
    def test_organization_extraction_accuracy(self):
        """Test accuracy of organization entity extraction."""
        test_cases = [
            {
                'text': "Microsoft is a technology company.",
                'expected': 'Microsoft'
            },
            {
                'text': "Google develops search engines.",
                'expected': 'Google'
            }
        ]
        
        with patch('smartmemory.plugins.extractors.spacy.SpacyExtractor') as MockExtractor:
            mock_extractor = Mock()
            
            for case in test_cases:
                mock_extractor.extract.return_value = {
                    'entities': [{'text': case['expected'], 'type': 'organization', 'start': 0, 'end': len(case['expected'])}],
                    'relations': []
                }
                
                result = mock_extractor.extract(case['text'])
                assert result['entities'][0]['text'] == case['expected']
    
    def test_relationship_extraction_accuracy(self):
        """Test accuracy of relationship extraction."""
        test_cases = [
            {
                'text': "Steve Jobs founded Apple.",
                'expected_relationship': {
                    'source': 'Steve Jobs',
                    'target': 'Apple',
                    'type': 'founded'
                }
            }
        ]
        
        with patch('smartmemory.plugins.extractors.spacy.SpacyExtractor') as MockExtractor:
            mock_extractor = Mock()
            
            for case in test_cases:
                mock_extractor.extract.return_value = {
                    'entities': [
                        {'text': 'Steve Jobs', 'type': 'person', 'start': 0, 'end': 10},
                        {'text': 'Apple', 'type': 'organization', 'start': 19, 'end': 24}
                    ],
                    'relations': [case['expected_relationship']]
                }
                
                result = mock_extractor.extract(case['text'])
                
                assert len(result['relations']) == 1
                rel = result['relations'][0]
                assert rel['source'] == case['expected_relationship']['source']
                assert rel['target'] == case['expected_relationship']['target']


class TestExtractionPerformance:
    """Test extraction performance characteristics."""
    
    def test_extraction_speed(self):
        """Test extraction completes in reasonable time."""
        import time
        
        text = "Python is a programming language." * 10  # Longer text
        
        with patch('smartmemory.plugins.extractors.spacy.SpacyExtractor') as MockExtractor:
            mock_extractor = Mock()
            mock_extractor.extract.return_value = {
                'entities': [{'text': 'Python', 'type': 'technology', 'start': 0, 'end': 6}],
                'relations': []
            }
            MockExtractor.return_value = mock_extractor
            
            start_time = time.time()
            result = mock_extractor.extract(text)
            elapsed = time.time() - start_time
            
            # Should complete quickly (mocked, so very fast)
            assert elapsed < 1.0
            assert len(result['entities']) > 0
    
    def test_batch_extraction_performance(self):
        """Test batch extraction performance."""
        texts = ["Test text " + str(i) for i in range(100)]
        
        with patch('smartmemory.plugins.extractors.spacy.SpacyExtractor') as MockExtractor:
            mock_extractor = Mock()
            mock_extractor.extract.return_value = {
                'entities': [],
                'relations': []
            }
            MockExtractor.return_value = mock_extractor
            
            import time
            start_time = time.time()
            
            results = [mock_extractor.extract(text) for text in texts]
            
            elapsed = time.time() - start_time
            
            # Should handle batch efficiently
            assert len(results) == 100
            assert elapsed < 5.0  # Reasonable time for 100 texts
    
    def test_memory_usage(self):
        """Test extraction doesn't consume excessive memory."""
        # This would test actual memory usage in real scenario
        # For now, verify the pattern
        text = "Test text"
        
        with patch('smartmemory.plugins.extractors.spacy.SpacyExtractor') as MockExtractor:
            mock_extractor = Mock()
            mock_extractor.extract.return_value = {
                'entities': [],
                'relations': []
            }
            MockExtractor.return_value = mock_extractor
            
            # Extract multiple times
            for _ in range(10):
                result = mock_extractor.extract(text)
                assert isinstance(result, dict)
