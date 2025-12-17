"""
Text chunking utilities for processing large texts.

chunk_size parameter, this module provides utilities
for splitting large texts into manageable chunks while preserving context.

Features:
- Character-based chunking with overlap
- Sentence-aware chunking (preserves sentence boundaries)
- Paragraph-aware chunking
- Configurable overlap for context preservation
"""

import logging
import re
from typing import List, Optional, Callable, Dict, Any

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = 5000,
    overlap: int = 200,
    strategy: str = "sentence",
    similarity_threshold: float = 0.85
) -> List[str]:
    """
    Split text into chunks for processing.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        strategy: Chunking strategy
            - "character": Simple character-based splitting
            - "sentence": Split on sentence boundaries
            - "paragraph": Split on paragraph boundaries
            - "recursive": Try paragraph → sentence → character in order
            - "markdown": Respect markdown headers and code blocks
            - "semantic": Use embeddings to find topic boundaries
        similarity_threshold: For semantic chunking, threshold for topic change
            
    Returns:
        List of text chunks
        
    Example:
        chunks = chunk_text(large_text, chunk_size=5000)
        for chunk in chunks:
            result = extractor.extract(chunk)
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    if strategy == "character":
        return _chunk_by_characters(text, chunk_size, overlap)
    elif strategy == "paragraph":
        return _chunk_by_paragraphs(text, chunk_size, overlap)
    elif strategy == "recursive":
        return _chunk_recursive(text, chunk_size, overlap)
    elif strategy == "markdown":
        return _chunk_by_markdown(text, chunk_size, overlap)
    elif strategy == "semantic":
        return _chunk_semantic(text, chunk_size, overlap, similarity_threshold)
    else:  # Default to sentence
        return _chunk_by_sentences(text, chunk_size, overlap)


def _chunk_by_characters(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple character-based chunking with overlap."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start forward, accounting for overlap
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


def _chunk_by_sentences(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Chunk text while preserving sentence boundaries.
    
    Tries to keep sentences intact when possible.
    """
    # Split into sentences
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # If single sentence exceeds chunk_size, split it
        if sentence_length > chunk_size:
            # Flush current chunk first
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Split long sentence by characters
            sub_chunks = _chunk_by_characters(sentence, chunk_size, overlap)
            chunks.extend(sub_chunks)
            continue
        
        # Check if adding this sentence exceeds chunk_size
        if current_length + sentence_length + 1 > chunk_size:
            # Save current chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            # Include last few sentences for context
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) + 1 <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s) + 1
                else:
                    break
            
            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
        else:
            current_chunk.append(sentence)
            current_length += sentence_length + 1
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def _chunk_by_paragraphs(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Chunk text while preserving paragraph boundaries.
    
    Paragraphs are identified by double newlines.
    """
    # Split into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_length = len(para)
        
        # If single paragraph exceeds chunk_size, use sentence chunking
        if para_length > chunk_size:
            # Flush current chunk first
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Split long paragraph by sentences
            sub_chunks = _chunk_by_sentences(para, chunk_size, overlap)
            chunks.extend(sub_chunks)
            continue
        
        # Check if adding this paragraph exceeds chunk_size
        if current_length + para_length + 2 > chunk_size:
            # Save current chunk
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            
            # Start new chunk (paragraphs don't overlap as cleanly)
            current_chunk = [para]
            current_length = para_length
        else:
            current_chunk.append(para)
            current_length += para_length + 2
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks


def _chunk_recursive(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Recursive chunking: try paragraph → sentence → character in order.
    
    Attempts to preserve larger semantic units first, falling back to
    smaller units when content exceeds chunk_size.
    """
    # First try paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    if len(paragraphs) == 1 and len(paragraphs[0]) > chunk_size:
        # Single paragraph too large, try sentences
        return _chunk_by_sentences(text, chunk_size, overlap)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_length = len(para)
        
        if para_length > chunk_size:
            # Paragraph too large - flush current and recurse
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Recursively chunk the large paragraph by sentences
            sub_chunks = _chunk_by_sentences(para, chunk_size, overlap)
            chunks.extend(sub_chunks)
        elif current_length + para_length + 2 > chunk_size:
            # Would exceed chunk size - start new chunk
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_length = para_length
        else:
            current_chunk.append(para)
            current_length += para_length + 2
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks


def _chunk_by_markdown(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Markdown-aware chunking that respects headers, code blocks, and lists.
    
    Preserves markdown structure by:
    - Never splitting inside code blocks
    - Preferring splits at header boundaries
    - Keeping list items together when possible
    """
    # Pattern to identify markdown sections
    header_pattern = r'(^#{1,6}\s+.*$)'
    code_block_pattern = r'(```[\s\S]*?```|`[^`]+`)'
    
    # Protect code blocks by replacing with placeholders
    code_blocks = []
    def save_code_block(match):
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks) - 1}__"
    
    protected_text = re.sub(code_block_pattern, save_code_block, text, flags=re.MULTILINE)
    
    # Split by headers
    sections = re.split(r'(^#{1,6}\s+.*$)', protected_text, flags=re.MULTILINE)
    
    chunks = []
    current_chunk = ""
    current_header = ""
    
    for section in sections:
        if not section.strip():
            continue
            
        # Check if this is a header
        if re.match(r'^#{1,6}\s+', section):
            # If current chunk is getting large, flush it
            if len(current_chunk) + len(section) > chunk_size and current_chunk:
                # Restore code blocks in current chunk
                for i, block in enumerate(code_blocks):
                    current_chunk = current_chunk.replace(f"__CODE_BLOCK_{i}__", block)
                chunks.append(current_chunk.strip())
                current_chunk = section
            else:
                current_chunk += section
            current_header = section
        else:
            # Regular content
            if len(current_chunk) + len(section) > chunk_size:
                if current_chunk:
                    # Restore code blocks
                    for i, block in enumerate(code_blocks):
                        current_chunk = current_chunk.replace(f"__CODE_BLOCK_{i}__", block)
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap (include current header)
                if current_header:
                    current_chunk = current_header + "\n" + section
                else:
                    current_chunk = section
            else:
                current_chunk += "\n" + section
    
    # Don't forget the last chunk
    if current_chunk:
        for i, block in enumerate(code_blocks):
            current_chunk = current_chunk.replace(f"__CODE_BLOCK_{i}__", block)
        chunks.append(current_chunk.strip())
    
    # If no markdown structure found, fall back to paragraph chunking
    if len(chunks) <= 1 and text.strip():
        return _chunk_by_paragraphs(text, chunk_size, overlap)
    
    return [c for c in chunks if c.strip()]


def _chunk_semantic(
    text: str,
    chunk_size: int,
    overlap: int,
    similarity_threshold: float = 0.85
) -> List[str]:
    """
    Semantic chunking using embeddings to detect topic boundaries.
    
    Splits text where consecutive sentences have low semantic similarity,
    indicating a topic change. Falls back to sentence chunking if
    embeddings are unavailable.
    
    Args:
        text: Input text
        chunk_size: Maximum chunk size
        overlap: Overlap between chunks (unused for semantic, kept for API consistency)
        similarity_threshold: Similarity below which we detect a boundary (0-1)
    """
    try:
        from smartmemory.plugins.embedding import create_embeddings
        import numpy as np
    except ImportError:
        logger.warning("Embeddings unavailable for semantic chunking, falling back to sentence")
        return _chunk_by_sentences(text, chunk_size, overlap)
    
    # Split into sentences first
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 2:
        return _chunk_by_sentences(text, chunk_size, overlap)
    
    # Generate embeddings for each sentence
    try:
        embeddings = []
        for sentence in sentences:
            if len(sentence) > 10:  # Skip very short sentences
                emb = create_embeddings(sentence)
                if emb is not None:
                    embeddings.append((sentence, np.array(emb)))
                else:
                    embeddings.append((sentence, None))
            else:
                embeddings.append((sentence, None))
    except Exception as e:
        logger.warning(f"Failed to generate embeddings: {e}, falling back to sentence chunking")
        return _chunk_by_sentences(text, chunk_size, overlap)
    
    # Find semantic boundaries by comparing consecutive sentences
    def cosine_similarity(a, b):
        if a is None or b is None:
            return 1.0  # Assume similar if embeddings unavailable
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return dot / (norm_a * norm_b)
    
    # Build chunks based on semantic boundaries
    chunks = []
    current_chunk = []
    current_length = 0
    
    for i, (sentence, embedding) in enumerate(embeddings):
        sentence_length = len(sentence)
        
        # Check if we should create a boundary
        should_split = False
        
        if i > 0 and embedding is not None:
            prev_sentence, prev_embedding = embeddings[i - 1]
            if prev_embedding is not None:
                similarity = cosine_similarity(prev_embedding, embedding)
                if similarity < similarity_threshold:
                    should_split = True
                    logger.debug(f"Semantic boundary detected: similarity={similarity:.3f}")
        
        # Force split if chunk is getting too large
        if current_length + sentence_length + 1 > chunk_size:
            should_split = True
        
        if should_split and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_length + 1
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def extract_with_chunking(
    text: str,
    extractor_fn: Callable[[str], Dict[str, Any]],
    chunk_size: int = 5000,
    overlap: int = 200,
    strategy: str = "sentence",
    cluster: bool = True,
    context: Optional[str] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None
) -> Dict[str, Any]:
    """
    Extract entities and relations from large text using chunking.
    
    This is the main entry point for processing large texts.
    
    Args:
        text: Input text (can be arbitrarily large)
        extractor_fn: Extraction function that takes text and returns
                     dict with 'entities' and 'relations'
        chunk_size: Maximum characters per chunk
        overlap: Characters to overlap between chunks
        strategy: Chunking strategy ("character", "sentence", "paragraph")
        cluster: Whether to cluster entities/relations after aggregation
        context: Optional domain context for clustering
        parallel: Whether to process chunks in parallel (default: True)
        max_workers: Maximum parallel workers (default: None = auto)
        
    Returns:
        Aggregated extraction result with:
        - entities: Merged and optionally clustered entities
        - relations: Merged and optionally clustered relations
        - chunk_count: Number of chunks processed
        - entity_clusters: (if cluster=True) Entity cluster mapping
        - edge_clusters: (if cluster=True) Relation cluster mapping
        
    Example:
        from smartmemory.plugins.extractors.gliner2 import GLiNER2Extractor
        
        extractor = GLiNER2Extractor()
        result = extract_with_chunking(
            large_document,
            extractor.extract,
            chunk_size=5000,
            cluster=True,
            context="Technical documentation",
            parallel=True
        )
    """
    from smartmemory.memory.pipeline.stages.clustering import aggregate_graphs
    
    # Check if chunking is needed
    if len(text) <= chunk_size:
        result = extractor_fn(text)
        result['chunk_count'] = 1
        if cluster:
            from smartmemory.memory.pipeline.stages.clustering import cluster_extraction_result
            result = cluster_extraction_result(result, context)
        return result
    
    # Chunk the text
    chunks = chunk_text(text, chunk_size, overlap, strategy)
    logger.info(f"Split text into {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    
    # Extract from chunks (parallel or sequential)
    if parallel and len(chunks) > 1:
        chunk_results = _extract_parallel(extractor_fn, chunks, max_workers)
    else:
        chunk_results = _extract_sequential(extractor_fn, chunks)
    
    if not chunk_results:
        return {"entities": [], "relations": [], "chunk_count": len(chunks), "error": "All chunks failed"}
    
    # Aggregate results
    aggregated = aggregate_graphs(chunk_results, cluster=cluster, context=context)
    aggregated['chunk_count'] = len(chunks)
    
    logger.info(f"Aggregated {len(chunk_results)} chunks: "
               f"{len(aggregated.get('entities', []))} entities, "
               f"{len(aggregated.get('relations', []))} relations")
    
    return aggregated


def _extract_sequential(
    extractor_fn: Callable[[str], Dict[str, Any]],
    chunks: List[str]
) -> List[Dict[str, Any]]:
    """Extract from chunks sequentially."""
    chunk_results = []
    for i, chunk in enumerate(chunks):
        try:
            result = extractor_fn(chunk)
            chunk_results.append(result)
            logger.debug(f"Chunk {i+1}/{len(chunks)}: {len(result.get('entities', []))} entities, "
                        f"{len(result.get('relations', []))} relations")
        except Exception as e:
            logger.warning(f"Extraction failed for chunk {i+1}: {e}")
            continue
    return chunk_results


def _extract_parallel(
    extractor_fn: Callable[[str], Dict[str, Any]],
    chunks: List[str],
    max_workers: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Extract from chunks in parallel using ThreadPoolExecutor.
    
    parallel chunk processing.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    chunk_results = []
    failed_count = 0
    
    # Default to reasonable number of workers
    if max_workers is None:
        import os
        max_workers = min(len(chunks), os.cpu_count() or 4, 8)
    
    logger.info(f"Processing {len(chunks)} chunks in parallel with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks
        future_to_idx = {
            executor.submit(extractor_fn, chunk): i 
            for i, chunk in enumerate(chunks)
        }
        
        # Collect results as they complete
        results_by_idx = {}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results_by_idx[idx] = result
                logger.debug(f"Chunk {idx+1}/{len(chunks)} completed: "
                           f"{len(result.get('entities', []))} entities")
            except Exception as e:
                logger.warning(f"Extraction failed for chunk {idx+1}: {e}")
                failed_count += 1
    
    # Sort results by original order
    for i in range(len(chunks)):
        if i in results_by_idx:
            chunk_results.append(results_by_idx[i])
    
    if failed_count > 0:
        logger.warning(f"{failed_count}/{len(chunks)} chunks failed")
    
    return chunk_results


class ChunkedExtractor:
    """
    Wrapper that adds chunking capability to any extractor.
    
    Usage:
        from smartmemory.plugins.extractors.gliner2 import GLiNER2Extractor
        
        base_extractor = GLiNER2Extractor()
        chunked = ChunkedExtractor(base_extractor, chunk_size=5000, parallel=True)
        
        # Now handles large texts automatically with parallel processing
        result = chunked.extract(very_large_document)
    """
    
    def __init__(
        self,
        extractor,
        chunk_size: int = 5000,
        overlap: int = 200,
        strategy: str = "sentence",
        auto_cluster: bool = True,
        parallel: bool = True,
        max_workers: Optional[int] = None
    ):
        """
        Initialize chunked extractor.
        
        Args:
            extractor: Base extractor with extract() method
            chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks
            strategy: Chunking strategy
            auto_cluster: Whether to automatically cluster after aggregation
            parallel: Whether to process chunks in parallel
            max_workers: Maximum parallel workers (None = auto)
        """
        self.extractor = extractor
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
        self.auto_cluster = auto_cluster
        self.parallel = parallel
        self.max_workers = max_workers
    
    def extract(
        self,
        text: str,
        context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract with automatic chunking for large texts.
        
        Args:
            text: Input text (any size)
            context: Optional domain context for clustering
            **kwargs: Additional arguments passed to base extractor
            
        Returns:
            Extraction result (chunked and aggregated if text is large)
        """
        # Define extraction function
        def extract_fn(chunk_text: str) -> Dict[str, Any]:
            return self.extractor.extract(chunk_text, **kwargs)
        
        return extract_with_chunking(
            text,
            extract_fn,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            strategy=self.strategy,
            cluster=self.auto_cluster,
            context=context,
            parallel=self.parallel,
            max_workers=self.max_workers
        )
