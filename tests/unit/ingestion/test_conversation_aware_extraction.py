"""
Unit tests for conversation-aware extraction in the ingestion pipeline.

Tests that:
1. The registry auto-selects conversation_aware_llm when context is provided
2. The extraction pipeline passes conversation context to the extractor
"""

import pytest


pytestmark = pytest.mark.unit
from unittest.mock import MagicMock, patch

from smartmemory.memory.ingestion.registry import IngestionRegistry
from smartmemory.memory.ingestion.extraction import ExtractionPipeline
from smartmemory.conversation.context import ConversationContext
from smartmemory.models.memory_item import MemoryItem


class TestConversationAwareExtractorSelection:
    """Test that the registry selects the right extractor based on context."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry."""
        return IngestionRegistry()

    def test_select_default_without_context(self, registry):
        """Without conversation context, select default extractor (groq)."""
        selected = registry.select_extractor_for_context(None)
        assert selected == 'groq'

    def test_select_default_with_empty_context(self, registry):
        """With empty conversation context, select default extractor."""
        selected = registry.select_extractor_for_context({})
        assert selected == 'groq'

    def test_select_default_with_empty_turn_history(self, registry):
        """With empty turn_history, select default extractor."""
        selected = registry.select_extractor_for_context({'turn_history': []})
        assert selected == 'groq'

    def test_select_conversation_aware_with_context(self, registry):
        """With conversation context containing turns, select conversation_aware_llm."""
        context = {
            'turn_history': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi there!'}
            ]
        }
        selected = registry.select_extractor_for_context(context)
        assert selected == 'conversation_aware_llm'

    def test_conversation_aware_extractor_registered(self, registry):
        """Verify conversation_aware_llm is registered by default."""
        assert registry.is_extractor_registered('conversation_aware_llm')
        assert 'conversation_aware_llm' in registry.list_extractors()


class TestExtractionPipelineWithConversationContext:
    """Test that the extraction pipeline properly handles conversation context."""

    @pytest.fixture
    def registry(self):
        """Create registry with mocked extractors."""
        registry = IngestionRegistry()
        return registry

    @pytest.fixture
    def observer(self):
        """Create mock observer."""
        return MagicMock()

    @pytest.fixture
    def pipeline(self, registry, observer):
        """Create extraction pipeline."""
        return ExtractionPipeline(registry, observer)

    def test_extract_passes_context_to_extractor(self, pipeline):
        """Test that conversation context is passed to the extractor."""
        # Create a mock extractor that accepts conversation_context
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {'entities': [], 'relations': []}

        # Replace the get_extractor to return our mock
        pipeline.registry.get_extractor = MagicMock(return_value=mock_extractor)
        pipeline.registry.select_extractor_for_context = MagicMock(return_value='conversation_aware_llm')

        # Create test item and context
        item = MemoryItem(content="Alice works at Google")
        conversation_context = {
            'conversation_id': 'test_conv',
            'turn_history': [
                {'role': 'user', 'content': 'Who is Alice?'}
            ],
            'entities': [{'name': 'Alice', 'type': 'person'}]
        }

        # Call extract_semantics with conversation context
        pipeline.extract_semantics(item, conversation_context=conversation_context)

        # Verify extractor was called
        assert mock_extractor.extract.called

    def test_extract_selects_conversation_aware_extractor(self, pipeline):
        """Test that extraction selects conversation_aware_llm when context is provided."""
        # Create a mock extractor
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {'entities': [], 'relations': []}

        # Spy on select_extractor_for_context
        original_select = pipeline.registry.select_extractor_for_context
        pipeline.registry.select_extractor_for_context = MagicMock(side_effect=original_select)
        pipeline.registry.get_extractor = MagicMock(return_value=mock_extractor)

        # Create test item and context
        item = MemoryItem(content="Test content")
        conversation_context = {
            'turn_history': [{'role': 'user', 'content': 'hello'}]
        }

        # Call extract_semantics
        pipeline.extract_semantics(item, conversation_context=conversation_context)

        # Verify select_extractor_for_context was called with the context
        pipeline.registry.select_extractor_for_context.assert_called_once_with(conversation_context)


class TestConversationContextConversion:
    """Test conversion between dict and ConversationContext."""

    def test_dict_to_conversation_context(self):
        """Test converting a dict to ConversationContext."""
        context_dict = {
            'conversation_id': 'test_conv',
            'participant_id': 'user_123',
            'turn_history': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi!'}
            ],
            'entities': [{'name': 'Alice', 'type': 'person'}],
            'topics': ['greeting']
        }

        ctx = ConversationContext.from_dict(context_dict)

        assert ctx.conversation_id == 'test_conv'
        assert ctx.participant_id == 'user_123'
        assert len(ctx.turn_history) == 2
        assert len(ctx.entities) == 1
        assert 'greeting' in ctx.topics

    def test_conversation_context_to_dict(self):
        """Test converting ConversationContext to dict."""
        ctx = ConversationContext(
            conversation_id='test_conv',
            participant_id='user_123'
        )
        ctx.turn_history.append({'role': 'user', 'content': 'Hello'})
        ctx.entities.append({'name': 'Alice', 'type': 'person'})

        context_dict = ctx.to_dict()

        assert context_dict['conversation_id'] == 'test_conv'
        assert context_dict['participant_id'] == 'user_123'
        assert len(context_dict['turn_history']) == 1
        assert len(context_dict['entities']) == 1

    def test_coreference_chains_in_context(self):
        """Test that coreference chains are properly stored and retrieved."""
        chains = [
            {'mentions': ['Apple Inc.', 'The company', 'it'], 'head': 'Apple Inc.'},
            {'mentions': ['Tim Cook', 'he', 'him'], 'head': 'Tim Cook'}
        ]

        ctx = ConversationContext(
            conversation_id='test_conv',
            coreference_chains=chains
        )

        assert len(ctx.coreference_chains) == 2
        assert ctx.coreference_chains[0]['head'] == 'Apple Inc.'
        assert 'it' in ctx.coreference_chains[0]['mentions']

    def test_coreference_chains_serialization(self):
        """Test coreference chains survive dict round-trip."""
        chains = [
            {'mentions': ['Google', 'The search giant', 'they'], 'head': 'Google'}
        ]

        ctx = ConversationContext(
            conversation_id='test_conv',
            coreference_chains=chains
        )

        # Round-trip through dict
        context_dict = ctx.to_dict()
        restored = ConversationContext.from_dict(context_dict)

        assert len(restored.coreference_chains) == 1
        assert restored.coreference_chains[0]['head'] == 'Google'
        assert 'The search giant' in restored.coreference_chains[0]['mentions']


class TestCoreferenceIntegration:
    """Test coreference chain integration with conversation-aware extraction."""

    def test_extractor_uses_coref_chains_for_resolution(self):
        """Test that the extractor uses fastcoref chains to resolve entities."""
        from smartmemory.plugins.extractors.conversation_aware_llm import ConversationAwareLLMExtractor

        extractor = ConversationAwareLLMExtractor()

        # Simulate entities that include pronouns
        entities = [
            {'name': 'The company', 'entity_type': 'organization'},
            {'name': 'quarterly results', 'entity_type': 'event'}
        ]

        # Provide coreference chains from fastcoref
        context = ConversationContext(
            coreference_chains=[
                {'mentions': ['Apple Inc.', 'The company', 'it'], 'head': 'Apple Inc.'}
            ]
        )

        resolved = extractor._resolve_coreferences(entities, context)

        # 'The company' should be resolved to 'Apple Inc.'
        resolved_names = [e['name'] for e in resolved]
        assert 'Apple Inc.' in resolved_names
        assert 'quarterly results' in resolved_names

        # Check resolution metadata
        apple_entity = next(e for e in resolved if e['name'] == 'Apple Inc.')
        assert apple_entity.get('resolved_from') == 'The company'
        assert apple_entity.get('resolution_source') == 'fastcoref'

    def test_extractor_fallback_to_heuristics(self):
        """Test heuristic resolution when no coref chains available."""
        from smartmemory.plugins.extractors.conversation_aware_llm import ConversationAwareLLMExtractor

        extractor = ConversationAwareLLMExtractor()

        entities = [
            {'name': 'it', 'entity_type': 'concept'}
        ]

        # Context with entity history but no coref chains
        context = ConversationContext(
            entities=[{'name': 'Machine Learning', 'type': 'concept'}]
        )

        resolved = extractor._resolve_coreferences(entities, context)

        # Should resolve 'it' using heuristic (last non-person entity)
        resolved_names = [e['name'] for e in resolved]
        assert 'Machine Learning' in resolved_names

    def test_context_text_includes_coref_chains(self):
        """Test that context text includes coreference information."""
        from smartmemory.plugins.extractors.conversation_aware_llm import ConversationAwareLLMExtractor

        extractor = ConversationAwareLLMExtractor()

        context = ConversationContext(
            coreference_chains=[
                {'mentions': ['OpenAI', 'The AI company', 'they'], 'head': 'OpenAI'}
            ],
            turn_history=[{'role': 'user', 'content': 'Tell me about OpenAI'}]
        )

        context_text = extractor._build_context_text(context)

        assert 'Coreference resolutions' in context_text
        assert 'OpenAI' in context_text
        assert 'The AI company' in context_text
