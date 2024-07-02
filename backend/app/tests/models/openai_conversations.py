from datetime import UTC, datetime, timezone
from uuid import UUID

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from backend.app.models.openai import (
    CONTENT_METADATA_MAP,
    AuthorRole,
    ContentType,
    Message,
    MessageAuthor,
    MessageCodeContent,
    MessageCodeMetadata,
    MessageContent,
    MessageMetadata,
    MessageTetherBrowsingDisplayContent,
    MessageTetherBrowsingDisplayMetadata,
    MessageTextContent,
    MessageTextMetadata,
)

# Hypothesis strategies
author_roles = st.sampled_from(list(AuthorRole))
content_types = st.sampled_from(list(ContentType))


@st.composite
def message_author_strategy(draw) -> dict[str, str]:
    return {
        "role": draw(author_roles),
        "name": draw(st.text(min_size=1, max_size=50)),
        "metadata": draw(st.none() | st.dictionaries(st.text(), st.text())),
    }


@st.composite
def message_content_strategy(draw) -> dict[str, str]:
    content_type = draw(content_types)
    if content_type == ContentType.TEXT:
        return {
            "content_type": content_type,
            "parts": draw(st.lists(st.text(), min_size=1, max_size=5)),
        }
    elif content_type == ContentType.CODE:
        return {
            "content_type": content_type,
            "language": draw(st.text(min_size=1, max_size=20)),
            "text": draw(st.text(min_size=1)),
        }
    # Add more content types as needed


@st.composite
def message_metadata_strategy(draw) -> dict[str, str]:
    content_type = draw(content_types)
    metadata_class = CONTENT_METADATA_MAP[content_type]
    return {
        "message_type": draw(st.none() | st.text()),
        "model_slug": draw(st.text(min_size=1, max_size=50)),
        "timestamp_": "absolute",
        # Add more common fields here
    }


# Hypothesis tests
@given(message_author_strategy())
def test_message_author_hypothesis(author_data) -> None:
    author = MessageAuthor(**author_data)
    assert author.role == author_data["role"]
    assert author.name == author_data["name"]


@given(message_content_strategy())
def test_message_content_hypothesis(content_data) -> None:
    content = MessageContent(**content_data)
    assert content.content_type == content_data["content_type"]
    if content.content_type == ContentType.TEXT:
        assert isinstance(content, MessageTextContent)
        assert content.parts == content_data["parts"]
    elif content.content_type == ContentType.CODE:
        assert isinstance(content, MessageCodeContent)
        assert content.language == content_data["language"]
        assert content.text == content_data["text"]


@given(message_metadata_strategy())
def test_message_metadata_hypothesis(metadata_data) -> None:
    metadata_class = CONTENT_METADATA_MAP[metadata_data.get("content_type", "text")]
    metadata = metadata_class(**metadata_data)
    assert metadata.model_slug == metadata_data["model_slug"]
    assert metadata.timestamp_ == metadata_data["timestamp_"]


# Traditional pytest tests
def test_message_with_tether_browsing_display() -> None:
    tether_content = {
        "content_type": "tether_browsing_display",
        "result": "Search results",
        "summary": "Summary of results",
        "assets": [],
        "tether_id": None,
    }
    tether_metadata = {
        "message_type": "tether_browsing_display",
        "model_slug": "gpt-4-browsing",
        "timestamp_": "absolute",
        "args": ["https://example.com"],
        "command": "browse",
        "status": "finished",
    }
    message_data = {
        "author": {"role": "assistant", "name": "AI"},
        "content": tether_content,
        "metadata": tether_metadata,
        "create_time": datetime.now(timezone.utc),
        "update_time": datetime.now(timezone.utc),
        "end_turn": True,
        "iden": UUID("12345678-1234-5678-1234-567812345678"),
        "recipient": "all",
        "status": "finished",
        "weight": 1.0,
    }
    message = Message(**message_data)
    assert isinstance(message.content, MessageTetherBrowsingDisplayContent)
    assert isinstance(message.metadata, MessageTetherBrowsingDisplayMetadata)
    assert message.content.result == "Search results"
    assert message.metadata.command == "browse"


def test_message_chain() -> None:
    # Test a chain of messages in a conversation
    messages = [
        Message(
            author=MessageAuthor(role=AuthorRole.USER, name="User"),
            content=MessageTextContent(content_type="text", parts=["Hello, AI!"]),
            metadata=MessageTextMetadata(
                message_type="text", model_slug="gpt-3.5-turbo", timestamp_="absolute"
            ),
            create_time=datetime.now(timezone.utc),
            update_time=datetime.now(timezone.utc),
            end_turn=True,
            iden=UUID("12345678-1234-5678-1234-567812345678"),
            recipient="all",
            status="finished",
            weight=1.0,
        ),
        Message(
            author=MessageAuthor(role=AuthorRole.ASSISTANT, name="AI"),
            content=MessageTextContent(
                content_type="text", parts=["Hello! How can I assist you today?"]
            ),
            metadata=MessageTextMetadata(
                message_type="text", model_slug="gpt-3.5-turbo", timestamp_="absolute"
            ),
            create_time=datetime.now(UTC),
            update_time=datetime.now(UTC),
            end_turn=True,
            iden=UUID("87654321-8765-4321-8765-432187654321"),
            recipient="all",
            status="finished",
            weight=1.0,
        ),
    ]
    assert len(messages) == 2
    assert messages[0].author.role == AuthorRole.USER
    assert messages[1].author.role == AuthorRole.ASSISTANT
    assert isinstance(messages[0].content, MessageTextContent)
    assert isinstance(messages[1].content, MessageTextContent)


def test_invalid_message_combination() -> None:
    with pytest.raises(ValidationError):
        Message(
            author=MessageAuthor(role=AuthorRole.USER, name="User"),
            content=MessageTextContent(content_type="text", parts=["Hello"]),
            metadata=MessageCodeMetadata(
                message_type="code", model_slug="gpt-3.5-turbo", timestamp_="absolute"
            ),
            create_time=datetime.now(timezone.utc),
            update_time=datetime.now(timezone.utc),
            end_turn=True,
            iden=UUID("12345678-1234-5678-1234-567812345678"),
            recipient="all",
            status="finished",
            weight=1.0,
        )


# Add more tests as needed
