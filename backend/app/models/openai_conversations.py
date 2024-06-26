"""
================================ MIT LICENSE =================================.

MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Copyright (c) 2024 Adam Poulemanos [modifications]

This module significantly modifies code written by Mohamed Cheikh Sidiya and licensed under the MIT license. Original code can be found at https://github.com/mohamed-chs/chatgpt-history-export-to-md.

Copyright (c) 2023 Mohamed Cheikh Sidiya [original]

To keep it simple, my modifications are also licensed under the MIT license.

================ _chat_models.py - WHAT CHANGED FROM CONVOVIZ ================

I added some Pydantic models and strengthened typing to improve validation and parsing, changed some class names that I thought were more intuitive, and refactored some code.
I did a lot of work to understand the metadata structure; that's all original. As you can see, it's very complex.

Thanks Mohamed for the original code! Saved me a lot of work. (Though the metadata structure was a nightmare... I know why you didn't touch it.)

=========================== PYDANTIC MODEL DIAGRAM ===========================

Major classes marked with <-----

ConversationHistoryJson  <----- # raw downloaded conversation history conversations.json
|
+-- Conversation  <----- # a single conversation, contains multiple ChatSessions
    |
    +-- ChatSession  <----- # a single chat session, contains multiple Message objects
        |
        +-- Message  <-----
            |
            +-- MessageAuthor
            |
            +-- MessageContent <-----
            |   |
            |   +-- MessageContentImagePart
            |       |
            |       +-- MessageContentImagePartMetadata
            |           |
            |           +-- DalleMetadata
            |
            +-- MessageMetadata  <-----
                |
                +-- Attachments
                |
                +-- FinishDetails
                |
                +-- UserContextMessageData
                |
                +-- AggregateResult
                |   |
                |   +-- InKernelException
                |   |
                |   +-- AggregateResultJupyterMessage
                |       |
                |       +-- AggregateResultJupyterMessageParentHeader
                |       |
                |       +-- AggregateResultJupyterMessageContent
                |           |
                |           +-- AggregateResultJupyterMessageContentData
                |
                +-- Citations
                |   |
                |   +-- CitationMetadata
                |       |
                |       +-- CitationMetadataExtra
                |
                +-- CiteMetadata
                |   |
                |   +-- CiteMetadataCitationFormat
                |   |
                |   +-- InsideCiteMetadataList
                |
                +-- JitPluginData
                    |
                    +-- JitFromClient
                    |   |
                    |   +-- JitFromClientTarget
                    |       |
                    |       +-- JitFromClientTargetData
                    |
                    +-- JitFromServer
                        |
                        +-- JitFromServerBody
                            |
                            +-- JitFromServerBodyActions
                            |   |
                            |   +-- JitFromServerBodyActionsTarget
                            |
                            +-- JitFromServerBodyParams


**Models below are in reverse order; top-level models at bottom of file**.

================================= BEGIN CODE =================================

"""

import mimetypes
import re
from pathlib import Path
from re import Pattern
from typing import Annotated, Any, ClassVar, Literal, Self
from uuid import UUID

from pydantic import (
    UUID4,
    AfterValidator,
    AnyHttpUrl,
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    Json,
    NonNegativeFloat,
    NonNegativeInt,
    PastDateTime,
    PositiveFloat,
    PositiveInt,
    PydanticValueError,
    field_validator,
)

# attempt to import most recent chat model from openai, but if it fails, use a recent literal

try:
    from openai.error import OpenAIError
    from openai.types.chat_model import ChatMetadataModel
except (ImportError, OpenAIError):
    ChatMetadataModel = Literal[
        "gpt-4o",
        "gpt-4o-2024-05-13",
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-0125-preview",
        "gpt-4-turbo-preview",
        "gpt-4-1106-preview",
        "gpt-4-vision-preview",
        "gpt-4",
        "gpt-4-0314",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0314",
        "gpt-4-32k-0613",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-16k-0613",
    ]


# ========================== CONVERSATION METADATA ===========================


class ChatMetadataModel(BaseModel):
    """Base class for all chat metadata models, we allow extra fields because we don't know what the future holds and metadata can be added at any time."""

    model_config = ConfigDict(
        extra="allow", title="ChatMetadataModel", arbitrary_types_allowed=True
    )


# ================== METADATA: JIT_PLUGIN_DATA ATTRIBUTE TREE ==================


class JitFromServerBodyActionsTarget(ChatMetadataModel):
    """ID and hash if applicable for a JIT server response to a plugin execution."""

    target_message_id: UUID4
    operation_hash: str | None = None


class JitFromServerBodyActions(ChatMetadataModel):
    """Actions for a JIT server response to a plugin execution."""

    typ: Literal["allow" | "always_allow" | "deny"] = Field(..., alias="type")

    allow: JitFromServerBodyActionsTarget | None = None
    always_allow: JitFromServerBodyActionsTarget | None = None
    deny: JitFromServerBodyActionsTarget | None = None
    name: Literal["allow" | "decline"] | None = None


class JitFromServerBodyParams(ChatMetadataModel):
    """Parameters for JIT server response to a plugin execution. If the plugin is a search plugin, the URL will be present. The query is the search query."""

    query: str
    url: AnyHttpUrl | None = None
    year_end: Annotated[PositiveInt, Field(pattern=r"^19[5-9]\d$|^20[0-2]\d$")] | None = None
    year_start: Annotated[PositiveInt, Field(pattern=r"^19[5-9]\d$|^20[0-2]\d$")] | None = None


class JitFromServerBody(ChatMetadataModel):
    """Information about the body of a JIT server response to a plugin execution."""

    actions: list[JitFromServerBodyActions]
    domain: AnyUrl
    is_consequential: bool
    method: Literal["get", "post"]
    privacy_policy: AnyHttpUrl

    params: JitFromServerBodyParams | None = None
    path: Path | None = None


class JitFromServer(ChatMetadataModel):
    """JIT server response to a plugin execution."""

    typ: Literal["preview"] | str = Field(..., alias="type")
    body: JitFromServerBody | None = None


class JitFromClientTargetData(ChatMetadataModel):
    """Information about the execution of a plugin."""

    typ: Literal["allow" | "always_allow" | "deny"] = Field(..., alias="type")
    operation_hash: str | None = None
    metadata: Any | None = None


class JitFromClientTarget(ChatMetadataModel):
    """target id and data for a plugin execution."""

    data: JitFromClientTargetData
    target_message_id: UUID4


class JitFromClient(ChatMetadataModel):
    """Information about a user action that triggered a Just-In-Time plugin."""

    user_action: list[JitFromClientTarget]


class JitPluginData(ChatMetadataModel):
    """Data for a Just-In-Time plugin. Appears in messages where any external (openai or third party) plugin was triggered by a user action. This can be code execution, internet search, database query, etc."""

    from_client: JitFromClient | None = None
    from_server: JitFromServer | None = None


# ================== METADATA: AGGREGATE_RESULT ATTRIBUTE TRE ==================


class InKernelException(ChatMetadataModel):
    """Information about an exception that occurred within a Jupyter kernel."""

    args_dict: dict[str, Any]
    args: list[str, ...] | None
    name: str  # name of the exception
    notes: list[str, ...] | None
    traceback: list[str]


class AggregateResultMessage(ChatMetadataModel):
    """Information about a message from a Jupyter notebook execution."""

    message_type: Literal["stream" | "image" | "timeout_interrupt"]
    sender: Literal["server"]
    time: PastDateTime

    image_payload: Any | None = None  # not observed with a value
    image_url: AnyUrl | None = None
    stream_name: Literal["stdout" | "stderr"] | None = None
    text: str | None = None
    timeout_triggered: float | None = None


class AggregateResultJupyterMessageContentData(ChatMetadataModel):
    """Actual content of a Jupyter message."""

    text_plain: str = Field(..., alias="text/plain")

    image_vnd_openai_fileservice_png: AnyUrl | None = Field(
        None, alias="image/vnd.openai.fileservice.png"
    )
    text_html: str | None = Field(None, alias="text/html")
    text_markdown: str | None = Field(None, alias="text/markdown")


class AggregateResultJupyterMessageContent(ChatMetadataModel):
    """Content of a Jupyter notebook execution message. If there was an error, ename, evalue, and traceback will be present."""

    data: AggregateResultJupyterMessageContentData | None = None
    ename: str | None = None
    evalue: str | None = None
    execution_state: Literal["busy" | "idle"] | None = None
    name: Literal["stdout" | "stderr"] | None = None
    text: str | None = None
    traceback: list[str] | None = None


class AggregateResultJupyterMessageParentHeader(ChatMetadataModel):
    """Header metadata for a Jupyter notebook execution."""

    msg_id: Annotated[
        str, Field(pattern=r"^[a-z0-9-]{33}_\d_\d$")
    ]  # uuid4 with a _ and two integers
    version: PositiveFloat


class AggregateResultJupyterMessage(ChatMetadataModel):
    """Message from a Jupyter notebook execution."""

    msg_type: Literal[
        "status"
        | "execute_input"
        | "execute_result"
        | "stream"
        | "error"
        | "@timeout"
        | "display_data"
    ]
    parent_header: AggregateResultJupyterMessageParentHeader
    content: AggregateResultJupyterMessageContent | None = None
    timeout: float | None = None


class AggregateResult(ChatMetadataModel):
    """Represents the results from a Jupyter notebook execution from the server."""

    code: Path
    end_time: PastDateTime
    final_expression_output: Path
    jupyter_messages: list[AggregateResultJupyterMessage, ...]
    messages: list[AggregateResultMessage, ...]
    run_id: UUID4
    start_time: PastDateTime
    status: Literal["cancelled" | "success"]
    update_time: PastDateTime

    in_kernel_exception: InKernelException | None = None
    system_exception: str | None = None
    timeout_triggered: PastDateTime | None = None


# ================== METADATA: CITE_METADATA (_cite_metadata) ==================
# metadata of the citations metadata


class CiteMetadataCitationFormat(ChatMetadataModel):
    """Metadata about the citation format. This is a literal value. If regex is present, it matches footnotes in the citation format."""

    name: Literal["tether_og"]
    regex: Pattern | None = None  # regex matches footnotes in the citation format


class InsideCiteMetadataList(ChatMetadataModel):
    """Information about individual citations in a message."""

    pub_date: PastDateTime | None
    text: str
    title: str
    url: AnyHttpUrl


class CiteMetadata(ChatMetadataModel):
    """Metadata about the citations in a message. Metadata metadata metadata... like inception, except cooler."""

    citation_format: CiteMetadataCitationFormat
    metadata_list: list[list[InsideCiteMetadataList], ...]
    original_query: str | None


# ===================== METADATA: CITATIONS ATTRIBUTE TREE =====================


class CitationMetadataExtra(ChatMetadataModel):
    """Additional citation metadata giving the index of the cited message."""

    cited_message_idx: PositiveInt
    evidence_text: str | Literal["source"]


class CitationMetadata(ChatMetadataModel):
    """Metdata associated with a citation."""

    pub_date: PastDateTime
    text: str
    title: str
    url: AnyHttpUrl
    extra: CitationMetadataExtra | None = None


class Citations(ChatMetadataModel):
    """Citations for a message. Present for research-oriented bots and in any web results."""

    start_ix: Annotated[PositiveInt, Field(ge=0)]
    end_ix: Annotated[PositiveInt, Field(lt=100000)]
    citation_format_type: Literal["tether_og", "tether_markdown"] | None = None
    metadata: list[CitationMetadata, ...] | None = None
    invalid_reason: str | None = None


# ==================== METADATA: OTHER METADATA ATTRIBUTES =====================

class Attachments(ChatMetadataModel):
    """Attachments to a `Message`."""

    iden: str = Field(..., alias="id")
    name: str

    fileTokenSize: int | None = None
    height: int | None = None
    mime: str | None = None
    url: AnyHttpUrl | None = None
    content_type: str | None = None
    size: PositiveInt | None = None

    _mime_types: ClassVar[dict[str, list[str]]] = {
        **mimetypes.types_map,
        **mimetypes.common_types} | {"application/toml": ["toml"], "text/markdown": ["md", "mk"], "application/yaml": ["yaml", "yml"]}

    @field_validator("mime")
    @classmethod
    def validate_mime(cls, value: str) -> str:
        """Validates a value as a mimetype."""
        if value not in cls._mime_types():
            raise ValueError(f"Invalid mime type: {value}")
        return value


class UserContextMessageData(ChatMetadataModel):
    """This is user-provided context and instructions that are sent with the message."""

    about_user_message: str | None = None
    about_model_message: str | None = None


class FinishDetails(ChatMetadataModel):
    """Details about message completion. Stop token is the model's token limit if it's hit. These are basically int literals, but they change by model so we'll keep it generic."""

    typ: str | Literal["stop"] | None = Field(None, alias="type")
    stop_tokens: list[PositiveInt] | None = None


class MessageMetadata(ChatMetadataModel):
    """
    Represents the metadata of a message within a conversation. All metadata fields are highly variable based on the tools and model used in a conversation. My subclasses are probably not comprehensive, but should go a long way towards helping anyone who wants to make sense of openai metadata.
    """
    _gpt_pattern = ClassVar[Pattern[str]] = re.compile(r"^gpt-[0-9]{1,2}[.]?5?[a-zA-Z0-9-]*$")
    _iata_code_pattern: ClassVar[Pattern[str]] = re.compile(
        r"(^[A-Fa-f0-9]{16})-([A-Z]{3}$)|(([A-Fa-f0-9]{16})-([A-Z]{4}$)")

    aggregate_result: AggregateResult | None = None
    args: list[AnyUrl | str | int | float] | None = None
    attachments: Attachments | None = None
    citations: list[Citations] | None = None
    cite_metadata: list[CiteMetadata] | None = Field(None, alias="_cite_metadata")
    command: str | None = None
    default_model_slug: Annotated[str, Field(pattern=_gpt_pattern)] | None = None
    finish_details: FinishDetails | None = None
    gizmo_id: str | None = None
    invoked_plugin: dict[str, Any] | None = None
    is_chat_system_message: bool | None = None
    is_complete: bool | None = None
    is_user_system_message: bool | None = None
    is_visually_hidden_from_conversation: bool | None = None
    jit_plugin_data: JitPluginData | None = None
    message_type: str | None = None
    model_slug: (
        Annotated[str, Field(pattern=_gpt_pattern, union_mode="left_to_right")] | str | None
    ) = None
    pad: Annotated[str, Field(pattern=r"^A*$")] | None = None
    parent_id: UUID4 | None = None
    rebase_system_message: bool | None = None
    request_id: str | None = None
    status: Literal["failed" | "finished"] | None = None
    timestamp_: Literal["absolute"] | None = None
    user_context_message_data: UserContextMessageData | None = None
    voice_mode_message: bool | None = None


    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str | None) -> UUID | tuple[str, str]:
        """Validates a value as a UUID4."""
        if value is None:
            return value
        if isinstance(value, str):
            if len(value) > 20:
                try:
                    return UUID(value, version=4)
                except ValueError as e:
                    raise ValueError(f"Invalid UUID4, value given: {value}") from e
            try:
                return cls._validate_iata_type(value)
            except ValueError as e:
                raise ValueError(f"Invalid request_id: {value}. Does not match UUID4 or tuple[hex_value, airport IATA code pattern]") from e

    @classmethod
    def _validate_iata_type(cls, value: str) -> str | "RequestIdValues" | UUID4:
        """Validate the hex-iata combined request_id type."""
        if rmatch := cls._iata_code_pattern.match(value):
            match rmatch.group(1, 2, 3, 4):
                case ((hex_code, iata_code, None, None) | (None, None, hex_code, iata_code)):
                    if located := cls.lookup_iata(iata_code):
                        return hex_code, located
        raise ValueError(f"Invalid request_id: {value}")

    @staticmethod
    def lookup_iata(s: str):
        """Lookup IATA code in a JSON file and return the location."""
        if 2 < len(s) < 5:
            return None
        import json
        try:
            with open("data/airports.json") as f:
                iatas = json.load(f)
            if iata := iatas.get(s):
                return ", ".join(iata)
        except FileNotFoundError:
            if match := re.match(r"([A-Z]{3,4})", s):
                return match[1]


# ================================ END METADATA ================================

# ================================= MESSAGE ==================================

"""
Message represents a single message in a Conversation. It's contained in a Session object.

"""


class ChatMessageFieldsModel(BaseModel):
    """A base class to hang a Config object on for message field models."""

    model_config = ConfigDict(extra="allow", smart_union=True)


class DalleMetadata(ChatMessageFieldsModel):
    """Primary field for metadata in a dalle message."""

    prompt: str  # this is the image generating prompt that ChatGPT gives to DALL-E
    seed: PositiveInt
    serialization_title: str

    edit_op: str | None
    gen_id: Annotated[str, AfterValidator(lambda x: len(x) == 16)] | None
    parent_gen_id: str | None


class MessageContentImagePartMetadata(ChatMessageFieldsModel):
    """Metadata associated with the `MessageContentImagePart` of a dalle message."""

    dalle: DalleMetadata
    sanitized: bool


class MessageContentImagePart(ChatMessageFieldsModel):
    """The part of a dalle message that contains the image pointer."""

    asset_pointer: Annotated[str, Field(pattern=r"file-service://file-[A-Za-z0-9]{24}")]
    content_type: Literal["image_asset_pointer"]
    height: PositiveInt
    size_bytes: NonNegativeInt
    width: PositiveInt

    fovea: int | None
    metadata: MessageContentImagePartMetadata | None


class MessageContent(ChatMessageFieldsModel):
    """Content of a message. This can be text, code, an image, etc."""

    content_type: (
        Literal[
            "text"
            | "code"
            | "execution_output"
            | "image"
            | "image_asset_pointer"
            | "multimodal_text"
            | "system_error"
            | "tether_browsing_display"
            | "tether_quote"
        ]
        | str
        | None
    )
    assets: list[Any] | None = None  # never observed a value for this
    language: str | None = Field(
        None,
        description="Language of the code block",
        examples=["python", "javascript", "json", "html", "css", "bash", "sql", "yaml"],
    )
    parts: list[list[str, ...]] | MessageContentImagePart | None = None
    text: str | None = None
    name: str | None = None
    url: AnyHttpUrl | None = None
    domain: AnyUrl | None = None
    title: str | None = None
    description: str | None = None
    tether_id: None = None
    result: str | None = None
    sanitized: bool | None = None


class MessageAuthor(ChatMessageFieldsModel):
    """Type of the `author` field in a `Message`."""

    metadata: dict[str, Any] | None
    name: str | Literal["browser" | "python"] | None
    role: Literal["assistant" | "user" | "system" | "tool"]

    @property
    def role_name(self, role_map: dict[str, str] | None = None) -> str:
        """Get the user-facing role name of the author."""
        role_map = role_map or {
            "assistant": "ChatGPT",
            "user": "Me",
            "system": "System",
            "tool": "Tool",
        }
        return role_map[self.role]


class ChatModel(BaseModel):
    """Base class for config object for others to inherit."""

    model_config = ConfigDict(smart_union=True)


class Message(ChatModel):
    """A single message (e.g. user to Assistant or Assistant to user) within a `ChatSession`."""

    author: MessageAuthor
    content: MessageContent
    create_time: PastDateTime
    end_turn: bool
    iden: UUID4 = Field(..., alias="id")
    metadata: MessageMetadata
    recipient: str | Literal["all" | "browser" | "python"]
    status: str
    update_time: PastDateTime
    weight: NonNegativeFloat

    @property
    def text(self) -> str:
        """Get the text content of the message."""
        if self.content.parts:
            return str(self.content.parts[0])
        if self.content.text:
            return self.code_block(self.content.text, self.content.language or "python")
        if self.content.result:
            return self.content.result

        # this error caught some hidden bugs in the data. need more of these
        err_msg = f"No valid content found in message: {self.id}"
        raise ValueError(err_msg)

    @staticmethod
    def close_code_blocks(text: str) -> str:
        """Ensure that all code blocks are closed."""
        open_code_block = False

        lines = text.split("\n")

        for line in lines:
            if line.startswith("```") and not open_code_block:
                open_code_block = True
                continue

            if line == "```" and open_code_block:
                open_code_block = False

        if open_code_block:
            text += "\n```"

        return text

    @staticmethod
    def code_block(text: str, lang: str = "python") -> str:  # from convoviz
        """Wrap the given string in a code block."""
        return f"```{lang}\n{text}\n```"


class ChatSession(ChatModel):
    """Class representing a single chat session in a conversation. OpenAI and convoviz call this a 'node', but I think 'session' is more intuitive. It's not a completely precise term, but it's close enough. [I do understand they are referring to tree nodes as a reference to the underlying data structure, but each essentially represents a chat session, so I'm calling it a session]."""

    iden: UUID4 = Field(..., alias="id")

    children: list[UUID4, ...] = Field(default_factory=list)
    message: Message | None = None
    parent: UUID4 | None = None

    _child_sessions: list[Self, ...] = []
    _parent_session: Self | None = None

    def add_child(self, session: Self) -> None:
        """Add a child to the session."""
        self._child_sessions.append(session)
        session._parent_session = self

    @classmethod
    def mapping(cls, mapping: dict[str, Self]) -> dict[str, Self]:
        """Return a dictionary of connected ChatSession objects, based on the mapping."""
        for session in mapping.values():
            session._child_sessions = []  # Ensure list is empty to avoid duplicates
            session._parent_session = None  # Ensure _parent_session is None

        # Connect sessions
        for session in mapping.values():
            for child_id in session.children:
                child_session = mapping[child_id]
                session.add_child(child_session)

        return mapping


class Conversation(ChatModel):
    """A Conversation is one or more ChatSessions (or nodes) and may be over a wide span of time if the user revisited a conversation."""

    moderation_results: list[Any] | None = []

    conversation_id: UUID4
    conversation_template_id: str | None
    create_time: PastDateTime
    current_node: UUID4
    gizmo_id: str | None
    iden: UUID4 = Field(..., alias="id")
    is_archived: bool
    mapping: dict[UUID4, ChatSession]
    plugin_ids: list[Annotated[str, Field(pattern=r"plugin-[a-f0-9-]{36}")]] | None
    safe_urls: list[AnyUrl, ...] | list[None]
    title: str
    update_time: PastDateTime

    @property
    def session_mapping(self) -> dict[str, ChatSession]:
        """Return a dictionary of connected ChatSession objects, based on the mapping."""
        return ChatSession.mapping(self.mapping)

    @property
    def _all_message_sessions(self) -> list[ChatSession]:
        """List of all sessions that have a message, including all branches."""
        return [session for session in self.session_mapping.values() if session.message]

    def _author_sessions(
        self, *authors: Literal["assistant" | "user" | "system" | "tool"]
    ) -> list[ChatSession]:
        """List of all sessions with the given author role (all branches)."""
        if not authors:
            authors = ("user",)
        return [
            session
            for session in self._all_message_sessions
            if session.message and session.message.author.role in authors
        ]

    @property
    def leaf_count(self) -> int:
        """Return the number of leaves in the conversation."""
        return sum(not session.children_sessions for session in self._all_message_sessions)

    @property
    def url(self) -> str:
        """Chat URL."""
        return f"https://chat.openai.com/c/{self.conversation_id}"

    @property
    def content_types(self) -> list[str]:
        """List of all content types in the conversation (all branches)."""
        return list({
            session.message.content.content_type
            for session in self._all_message_sessions
            if session.message
        })

    def message_count(self, *authors: Literal["assistant" | "user" | "system" | "tool"]) -> int:
        """Return the number of 'user' and 'assistant' messages (all branches)."""
        if not authors:
            authors = ("user",)
        return len(self._author_sessions(*authors))

    @property
    def model(self) -> str | None:
        """ChatGPT model used for the conversation."""
        assistant_sessions: list[ChatSession] = self._author_sessions("assistant")
        if not assistant_sessions:
            return None

        message = assistant_sessions[0].message

        return message.metadata.model_slug if message else None

    @property
    def plugins(self) -> list[str]:
        """List of all ChatGPT plugins used in the conversation."""
        return list({
            session.message.metadata.invoked_plugin["namespace"]
            for session in self._author_sessions("tool")
            if session.message and session.message.metadata.invoked_plugin
        })

    @property
    def custom_instructions(self) -> dict[str, str]:
        """Return custom instructions used for the conversation."""
        system_sessions = self._author_sessions("system")
        if len(system_sessions) < 2:
            return {}

        context_message = system_sessions[1].message
        if context_message and context_message.metadata.is_user_system_message:
            return context_message.metadata.user_context_message_data or {}

        return {}

    def timestamps(
        self, *authors: Literal["assistant" | "user" | "system" | "tool"]
    ) -> list[float]:
        """List of all message timestamps from the given author role (all branches).

        Useful for generating time graphs.
        """
        if not authors:
            authors = ("user",)
        return [
            session.message.create_time.timestamp()
            for session in self._author_sessions(*authors)
            if session.message and session.message.create_time
        ]


class ConversationHistoryJson(ChatModel):
    """The raw json file, conversations.json, in an openai downloaded chat history."""

    json_obj: Json[list[Conversation]]
