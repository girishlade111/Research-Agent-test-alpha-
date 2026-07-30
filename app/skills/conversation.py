"""Multi-turn conversation management skill.

Stores conversation history per user+project and provides
query rewriting with pronoun resolution.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class Conversation:
    """A multi-turn conversation."""

    id: str
    user_id: str
    project_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class ConversationManager:
    """Manages multi-turn conversations per user+project."""

    def __init__(self) -> None:
        self.conversations: dict[str, Conversation] = {}

    def start_conversation(self, user_id: str, project_id: str) -> str:
        """Start a new conversation and return its ID.

        Args:
            user_id: The user starting the conversation.
            project_id: The project context.

        Returns:
            The conversation ID.
        """
        conv_id = str(uuid.uuid4())
        self.conversations[conv_id] = Conversation(
            id=conv_id,
            user_id=user_id,
            project_id=project_id,
        )
        return conv_id

    def add_turn(self, conv_id: str, role: str, content: str) -> None:
        """Add a turn to an existing conversation.

        Args:
            conv_id: Conversation ID.
            role: 'user' or 'assistant'.
            content: The message content.

        Raises:
            KeyError: If conversation not found.
        """
        conv = self.conversations.get(conv_id)
        if conv is None:
            raise KeyError(f"Conversation '{conv_id}' not found")
        conv.turns.append(ConversationTurn(role=role, content=content))

    def get_context(self, conv_id: str, max_turns: int = 10) -> list[dict[str, str]]:
        """Get recent conversation turns as context.

        Args:
            conv_id: Conversation ID.
            max_turns: Maximum number of recent turns.

        Returns:
            List of dicts with 'role' and 'content' keys.

        Raises:
            KeyError: If conversation not found.
        """
        conv = self.conversations.get(conv_id)
        if conv is None:
            raise KeyError(f"Conversation '{conv_id}' not found")
        recent = conv.turns[-max_turns:]
        return [{"role": t.role, "content": t.content} for t in recent]

    def get_conversation(self, conv_id: str) -> Conversation | None:
        """Get a conversation by ID."""
        return self.conversations.get(conv_id)

    def get_history(self, conv_id: str) -> list[dict[str, Any]]:
        """Get full conversation history formatted for API response.

        Args:
            conv_id: Conversation ID.

        Returns:
            List of turn dicts with role, content, and timestamp.

        Raises:
            KeyError: If conversation not found.
        """
        conv = self.conversations.get(conv_id)
        if conv is None:
            raise KeyError(f"Conversation '{conv_id}' not found")
        return [
            {
                "role": t.role,
                "content": t.content,
                "timestamp": t.timestamp.isoformat() + "Z",
            }
            for t in conv.turns
        ]


def rewrite_query(original_query: str, conversation_history: list[dict[str, str]]) -> str:
    """Rewrite a query by expanding pronouns using conversation context.

    If the query contains pronouns like 'it', 'this', 'that', 'they',
    'them', 'those', prepend key nouns from the previous turn.

    Args:
        original_query: The user's new query.
        conversation_history: List of previous turns.

    Returns:
        Rewritten query with expanded context, or original if no expansion needed.
    """
    if not conversation_history:
        return original_query

    # Check for pronouns that might need resolution
    pronouns = {"it", "this", "that", "they", "them", "those", "these", "its", "their"}
    query_lower = original_query.lower()
    query_words = set(re.findall(r"\b\w+\b", query_lower))

    needs_rewrite = bool(query_words & pronouns)
    if not needs_rewrite:
        return original_query

    # Get key nouns from previous turns (look at last 2 turns)
    recent_turns = conversation_history[-2:]
    context_text = " ".join(t["content"] for t in recent_turns)

    # Extract potential subject nouns (capitalized words, multi-word terms)
    nouns: list[str] = []
    # Proper nouns
    proper = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", context_text)
    nouns.extend(proper[:3])

    # Key terms (non-stopword words that appear significant)
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should",
        "can", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "but", "and", "or", "if", "not", "no", "what", "how", "when",
        "where", "why", "who", "which", "that", "this", "it", "its", "they",
    }
    words = re.findall(r"\b[a-z]{3,}\b", context_text.lower())
    key_terms = [w for w in words if w not in stopwords]
    # Get most common terms from context
    from collections import Counter
    term_counts = Counter(key_terms)
    top_terms = [t for t, _ in term_counts.most_common(3)]

    # Build context prefix
    context_parts = nouns + [t for t in top_terms if t not in [n.lower() for n in nouns]]
    if context_parts:
        context_prefix = "Regarding " + ", ".join(context_parts[:3]) + ": "
        return context_prefix + original_query

    return original_query
