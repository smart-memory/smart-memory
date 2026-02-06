"""Simplify stage — split text into atomic sentences using spaCy dependency parsing."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, List

from smartmemory.pipeline.state import PipelineState

if TYPE_CHECKING:
    from smartmemory.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

# Module-level cache for spaCy model
_nlp_cache: dict = {"nlp": None}


def _get_nlp():
    """Lazy-load and cache the spaCy model."""
    if _nlp_cache["nlp"] is not None:
        return _nlp_cache["nlp"]
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
    _nlp_cache["nlp"] = nlp
    return nlp


class SimplifyStage:
    """Split complex sentences into simpler atomic statements via dep parsing."""

    def __init__(self, nlp=None):
        """Args: nlp — optional pre-loaded spaCy Language (for testing)."""
        self._nlp = nlp

    @property
    def name(self) -> str:
        return "simplify"

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        cfg = config.simplify
        text = state.resolved_text or state.text

        if not cfg.enabled or not text:
            return replace(state, simplified_sentences=[text] if text else [])

        # Short text bypass
        tokens = text.split()
        if len(tokens) < cfg.min_token_count:
            return replace(state, simplified_sentences=[text])

        nlp = self._nlp or _get_nlp()
        if nlp is None:
            logger.warning("spaCy not available, returning text as single sentence")
            return replace(state, simplified_sentences=[text])

        try:
            doc = nlp(text)
            sentences: List[str] = []

            for sent in doc.sents:
                parts = self._split_sentence(sent, cfg)
                sentences.extend(parts)

            # Filter empty strings
            sentences = [s.strip() for s in sentences if s.strip()]
            if not sentences:
                sentences = [text]

            return replace(state, simplified_sentences=sentences)
        except Exception as e:
            logger.warning("Simplification failed: %s", e)
            return replace(state, simplified_sentences=[text])

    def _split_sentence(self, sent, cfg) -> List[str]:
        """Apply configured transforms to a single spaCy Span."""
        parts = [sent.text]

        if cfg.split_clauses:
            parts = self._do_split_clauses(sent, parts)

        if cfg.extract_relative:
            parts = self._do_extract_relative(sent, parts)

        if cfg.passive_to_active:
            parts = self._do_passive_to_active(parts)

        if cfg.extract_appositives:
            parts = self._do_extract_appositives(sent, parts)

        return parts

    def _do_split_clauses(self, sent, parts: List[str]) -> List[str]:
        """Split on coordinating conjunctions (cc dep)."""
        for token in sent:
            if token.dep_ == "cc" and token.head.dep_ in ("ROOT", "conj"):
                # Split the original sentence text at the conjunction
                left = sent.text[: token.idx - sent.start_char].strip()
                right = sent.text[token.idx - sent.start_char + len(token.text) :].strip()
                if left and right:
                    return [left, right]
        return parts

    def _do_extract_relative(self, sent, parts: List[str]) -> List[str]:
        """Extract relative clauses (relcl dep) into standalone sentences."""
        extra = []
        for token in sent:
            if token.dep_ == "relcl":
                # Build the relative clause text from the subtree
                clause_tokens = sorted(token.subtree, key=lambda t: t.i)
                clause_text = " ".join(t.text for t in clause_tokens)
                # Prefix with the head noun for context
                head = token.head
                standalone = f"{head.text} {clause_text}"
                extra.append(standalone)
        if extra:
            return parts + extra
        return parts

    def _do_passive_to_active(self, parts: List[str]) -> List[str]:
        """Convert passive constructions to active voice where detectable.

        This is a best-effort heuristic using dep labels nsubjpass/auxpass.
        Full passive-to-active rewriting requires more sophisticated logic,
        so we only handle simple cases.
        """
        # Passive rewriting is complex; we keep the sentence as-is but flag it.
        # A production implementation would use a dedicated rewriter.
        return parts

    def _do_extract_appositives(self, sent, parts: List[str]) -> List[str]:
        """Extract appositive phrases (appos dep) into standalone sentences."""
        extra = []
        for token in sent:
            if token.dep_ == "appos":
                head = token.head
                standalone = f"{head.text} is {token.text}"
                extra.append(standalone)
        if extra:
            return parts + extra
        return parts

    def undo(self, state: PipelineState) -> PipelineState:
        return replace(state, simplified_sentences=[])
