"""
Candidate Namer for CFS-3b Recommendation Engine.

Auto-generates names and descriptions for procedure candidates
based on cluster content analysis.
"""

import logging
from collections import Counter
from typing import List, Tuple

from smartmemory.models.memory_item import MemoryItem

logger = logging.getLogger(__name__)


class CandidateNamer:
    """
    Auto-generates names and descriptions for procedure candidates.

    Analyzes cluster content to:
    - Extract common entities and topics
    - Generate concise, descriptive names (max 50 chars)
    - Generate informative descriptions (max 200 chars)
    """

    # Common stop words to filter out
    STOP_WORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "i",
        "you",
        "we",
        "they",
        "he",
        "she",
        "my",
        "your",
        "our",
        "their",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "also",
    }

    # Technical terms to prioritize
    TECHNICAL_TERMS = {
        "api",
        "database",
        "query",
        "function",
        "method",
        "class",
        "module",
        "service",
        "endpoint",
        "authentication",
        "authorization",
        "error",
        "exception",
        "validation",
        "parsing",
        "processing",
        "handling",
        "request",
        "response",
        "configuration",
        "deployment",
        "testing",
        "caching",
        "optimization",
        "monitoring",
        "logging",
        "debugging",
        "integration",
        "migration",
        "transformation",
        "extraction",
        "workflow",
    }

    # Action verbs for name generation
    ACTION_VERBS = {
        "handle",
        "process",
        "validate",
        "parse",
        "extract",
        "transform",
        "create",
        "update",
        "delete",
        "fetch",
        "send",
        "receive",
        "execute",
        "configure",
        "initialize",
        "authenticate",
        "authorize",
        "cache",
        "optimize",
        "monitor",
        "log",
        "debug",
        "test",
        "deploy",
        "migrate",
    }

    def __init__(self, max_name_length: int = 50, max_description_length: int = 200):
        self.max_name_length = max_name_length
        self.max_description_length = max_description_length

    def generate_name_and_description(self, cluster: List[MemoryItem]) -> Tuple[str, str]:
        """
        Generate a suggested name and description for a cluster.

        Args:
            cluster: List of similar memory items

        Returns:
            Tuple of (suggested_name, suggested_description)
        """
        if not cluster:
            return "Unnamed Pattern", "No description available"

        # Extract key terms from all items
        all_terms = self._extract_key_terms(cluster)

        # Extract skills and tools from metadata
        skills, tools = self._extract_skills_and_tools(cluster)

        # Generate name
        name = self._generate_name(all_terms, skills, tools)

        # Generate description
        description = self._generate_description(cluster, all_terms, skills, tools)

        return name, description

    def _extract_key_terms(self, cluster: List[MemoryItem]) -> Counter:
        """Extract and count key terms from cluster content."""
        term_counter = Counter()

        for item in cluster:
            content = getattr(item, "content", "") or ""

            # Tokenize and clean
            words = content.lower().split()
            words = [w.strip(".,!?;:()[]{}\"'-") for w in words if len(w) > 2]

            # Filter stop words
            words = [w for w in words if w not in self.STOP_WORDS]

            # Add to counter
            term_counter.update(words)

        return term_counter

    def _extract_skills_and_tools(self, cluster: List[MemoryItem]) -> Tuple[Counter, Counter]:
        """Extract skills and tools from cluster metadata."""
        skills = Counter()
        tools = Counter()

        for item in cluster:
            metadata = getattr(item, "metadata", {}) or {}
            skills.update(metadata.get("skills", []))
            tools.update(metadata.get("tools", []))

        return skills, tools

    def _generate_name(self, terms: Counter, skills: Counter, tools: Counter) -> str:
        """Generate a concise name for the pattern."""
        # Prioritize technical terms
        technical_found = [(term, count) for term, count in terms.most_common(20) if term in self.TECHNICAL_TERMS]

        # Find action verbs
        action_found = [(term, count) for term, count in terms.most_common(20) if term in self.ACTION_VERBS]

        # Build name components
        components = []

        # Add most common skill/tool if available
        if skills:
            top_skill = skills.most_common(1)[0][0]
            components.append(self._title_case(top_skill))
        elif tools:
            top_tool = tools.most_common(1)[0][0]
            components.append(self._title_case(top_tool))

        # Add action verb
        if action_found:
            verb = action_found[0][0]
            components.append(self._title_case(verb))
        else:
            # Default action
            components.append("Processing")

        # Add technical term
        if technical_found:
            tech_term = technical_found[0][0]
            if tech_term not in [c.lower() for c in components]:
                components.append(self._title_case(tech_term))
        elif terms:
            # Use most common non-stop word
            top_term = terms.most_common(1)[0][0]
            if top_term not in [c.lower() for c in components]:
                components.append(self._title_case(top_term))

        # Construct name
        if len(components) >= 2:
            name = f"{components[0]} {components[1]}"
            if len(components) >= 3:
                name += f" {components[2]}"
        else:
            name = components[0] if components else "Detected Pattern"

        # Add "Pattern" suffix if name is too short or generic
        if len(name) < 15 and "Pattern" not in name:
            name += " Pattern"

        # Truncate if too long
        if len(name) > self.max_name_length:
            name = name[: self.max_name_length - 3] + "..."

        return name

    def _generate_description(self, cluster: List[MemoryItem], terms: Counter, skills: Counter, tools: Counter) -> str:
        """Generate a descriptive summary for the pattern."""
        # Get representative content from first item
        representative = cluster[0]
        content_preview = (getattr(representative, "content", "") or "")[:100]

        # Build description
        parts = []

        # Start with pattern type
        item_count = len(cluster)
        parts.append(f"Pattern detected in {item_count} similar items")

        # Mention skills/tools if present
        if skills:
            top_skills = [s for s, _ in skills.most_common(3)]
            parts.append(f"involving {', '.join(top_skills)}")
        elif tools:
            top_tools = [t for t, _ in tools.most_common(3)]
            parts.append(f"using {', '.join(top_tools)}")

        # Add content preview
        if content_preview:
            # Clean preview
            preview = content_preview.strip()
            if len(preview) > 80:
                preview = preview[:77] + "..."
            parts.append(f'Example: "{preview}"')

        # Join parts
        description = ". ".join(parts)

        # Truncate if too long
        if len(description) > self.max_description_length:
            description = description[: self.max_description_length - 3] + "..."

        return description

    def get_common_skills(self, cluster: List[MemoryItem], limit: int = 5) -> List[str]:
        """Get the most common skills across cluster items."""
        skills = Counter()
        for item in cluster:
            metadata = getattr(item, "metadata", {}) or {}
            skills.update(metadata.get("skills", []))
        return [skill for skill, _ in skills.most_common(limit)]

    def get_common_tools(self, cluster: List[MemoryItem], limit: int = 5) -> List[str]:
        """Get the most common tools across cluster items."""
        tools = Counter()
        for item in cluster:
            metadata = getattr(item, "metadata", {}) or {}
            tools.update(metadata.get("tools", []))
        return [tool for tool, _ in tools.most_common(limit)]

    @staticmethod
    def _title_case(word: str) -> str:
        """Convert word to title case."""
        return word.capitalize() if word else ""
