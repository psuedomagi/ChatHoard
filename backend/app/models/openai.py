"""
Some elements of this module are based on the convoviz library by Mohamed Cheikh Sidiya, which is licensed under the MIT license. Very little original code remains, but it was a good starting point for this project. Elements with some remaining code are marked #MCS-MIT. The MIT license notice is at the end of this file.
Copyright (c) 2023 Mohamed Cheikh Sidiya
repository:    https://github.com/mohamed-chs/chatgpt-history-export-to-md
license [MIT]: https://github.com/mohamed-chs/chatgpt-history-export-to-md/LICENSE


Copyright (c) 2024 Adam Poulemanos, licensed under [Apache 2.0](https://github.com/seekinginfiniteloop/ChatHoard/LICENSE.md).

=========================== PYDANTIC MODEL DIAGRAM ===========================

# Pydantic Model Relationships

See chart in [openai.mmd](https://github.com/seekinginfiniteloop/ChatHoard/docs/openai.mmd) for a diagram  of the relationships between the models... it's a bit complex.

**Models below are in roughly reverse order, grouped by relationship; top-level models at bottom of file**.

================================= BEGIN CODE =================================

"""

import contextlib
import json
import mimetypes
import re
from collections import defaultdict
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from re import Pattern
from typing import Annotated, Any, ClassVar, ForwardRef, Hashable, Literal, Self, Union
from zipfile import ZipFile

from pydantic import (
    UUID4,
    AfterValidator,
    AnyHttpUrl,
    AnyUrl,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    GetCoreSchemaHandler,
    HttpUrl,
    Json,
    NonNegativeFloat,
    NonNegativeInt,
    PastDatetime,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
    Tag,
    ValidationError,
    field_validator,
    root_validator,
)
from pydantic.main import IncEx
from pydantic_core import core_schema
from validators import hostname as is_hostname

# Constants

# For the mimetype field in Attachments
YAML_MIME = "application/yaml"
MARKDOWN_MIME = "text/markdown"
TOML_MIME = "application/toml"

# UI names for author roles
UI_ASSISTANT = "ChatGPT"
UI_USER = "me"
UI_SYSTEM = "system"
UI_TOOL = "tool"

# ================================ CUSTOM TYPES ================================
"""Custom types for Pydantic models."""

# Literals that describe the kind of Message content in a chat's Message object.
type ContentType = Literal[  # type: ignore[valid-type]
    "text",
    "code",
    "tether_browsing_display",
    "tether_quote",
    "execution_output",
    "system_error",
    "multimodal_text",
]

type Hostname = Annotated[  # type: ignore[valid-type]
    str,
    AfterValidator(is_hostname),
]

"""Main MessageContent type with discriminator"""
type MessageContent = Annotated[  # type: ignore[valid-type]
    Union[
        Annotated[MessageTextContent, Tag("TextContent")],
        Annotated[MessageCodeContent, Tag("CodeContent")],
        Annotated[
            MessageTetherBrowsingDisplayContent, Tag("TetherBrowsingDisplayContent")
        ],
        Annotated[MessageTetherQuoteContent, Tag("TetherQuoteContent")],
        Annotated[MessageExecutionOutputContent, Tag("ExecutionOutputContent")],
        Annotated[MessageSystemErrorContent, Tag("SystemErrorContent")],
        Annotated[MessageMultimodalTextContent, Tag("MultimodalTextContent")],
    ],
    Discriminator("content_type"),
    Tag("MessageContentDiscriminator"),
]

type MessageMetadata = Annotated[  # type: ignore[valid-type]
    Union[
        Annotated[MessageTextMetadata, Tag("TextMetadata")],
        Annotated[MessageCodeMetadata, Tag("CodeMetadata")],
        Annotated[
            MessageTetherBrowsingDisplayMetadata, Tag("TetherBrowsingDisplayMetadata")
        ],
        Annotated[MessageTetherQuoteMetadata, Tag("TetherQuoteMetadata")],
        Annotated[MessageExecutionOutputMetadata, Tag("ExecutionOutputMetadata")],
        Annotated[MessageSystemErrorMetadata, Tag("SystemErrorMetadata")],
        Annotated[MessageMultimodalTextMetadata, Tag("MultimodalTextMetadata")],
    ],
    Discriminator("content_type"),
    Tag("MessageMetadataTypes"),
]


class RequestIdType:
    """A custom type for request_id. This is a 16-character hex code followed by a 3-4 character IATA code."""

    _iata_code_pattern: ClassVar[Pattern[str]] = re.compile(
        r"(^[A-Fa-f0-9]{16})-([A-Z]{3,4}$)"
    )

    def __init__(self, value: str) -> None:
        self.value = value
        self._hex_code, self._location = self._parse_value(value)

    @classmethod
    def _parse_value(cls, value: str) -> tuple[str, tuple[str, ...]]:
        if match := cls._iata_code_pattern.match(value):
            hex_code, iata_code = match.group(1), match.group(2)
            if location := cls.lookup_iata(iata_code):
                return hex_code, location
        raise ValueError(f"Invalid request_id: {value}")

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.union_schema([
            core_schema.is_instance_schema(cls),
            core_schema.no_info_plain_validator_function(cls.validate),
        ])

    @classmethod
    def validate(cls, v: Any) -> "RequestIdType":
        if isinstance(v, cls):
            return v
        if isinstance(v, str):
            return cls(v)
        raise ValueError(f"Invalid type for RequestIdType: {type(v)}")

    @property
    def hex_code(self) -> str:
        """Return the hex code of the request ID."""
        return self._hex_code

    @property
    def location(self) -> tuple[str, ...]:
        """Return the location of the request ID."""
        return self._location

    @staticmethod
    def lookup_iata(s: str) -> tuple[str, ...]:
        """Look up an IATA code in a JSON file to get the location."""
        if 2 < len(s) < 5:
            with contextlib.suppress(FileNotFoundError):
                with open("data/airports.json") as f:
                    iatas = json.load(f)
                if iata := iatas.get(s):
                    return tuple(iata)
        return (s, "unknown_location")

    def __repr__(self):
        return f"RequestIdType(hex_code={self.hex_code}, location={self.location})"

    def __str__(self):
        return f"{self.hex_code} - {self.location}"


class AuthorRole(StrEnum):
    """Enum for author roles in a message. See [`MessageAuthor.role`](#message_author)."""

    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"
    TOOL = "tool"

    @property
    def ui_name(self) -> str:
        """Return the UI name of the role."""
        return {
            "assistant": UI_ASSISTANT,
            "user": UI_USER,
            "system": UI_SYSTEM,
            "tool": UI_TOOL,
        }[self.value]


# <a name="basemodels"></a>

# ================================ BASE MODELS ================================

"""Pydantic foundational BaseModel classes for all models."""

class ChatModel(BaseModel):
    """Base for config object for others to inherit."""

    model_config = ConfigDict(
        title="ChatModel",
        cache_strings="keys",
        hide_input_in_errors=True,
        allow_inf_nan=False,
    )

class FlexibleModel(BaseModel):
    """Base model for models to inherit where flexible validation is needed. We allow extra fields, but also implement custom validation handling that allows validation failures, dumping errors to logs. We will use this to get better data on fields like Message `content` and `metadata` where keys, values and types are likely to change regularly and could be highly variable based on how a user interacts with the model."""

    model_config = ConfigDict(
        extra="allow",
        title="FlexibleModel",
        cache_strings="keys",
        allow_inf_nan=False,
    )
    @root_validator(pre=True)
    def log_validation_errors(cls, values: dict[str, Any]) -> dict[str, Any]:
        validated_values = {}
        errors = []

        for key, value in values.items():
            try:
                validated_values[key] = cls.model_validate(cls.model_fields[key])
            except ValidationError as e:
                errors.append((key, e.errors()))
                validated_values[key] = value  # Keep original value to not lose data

        if errors:
            for error in errors:
                print(f"Validation error in {error[0]}: {error[1]}")

        return validated_values


class MessageContentModel(ChatModel):
    """Base for all message content types, which are discriminated by the `content_type` field in `Message.content`"""

    model_config = ConfigDict(
        title="MessageContentModel",
        cache_strings="keys",
        hide_input_in_errors=True,
        allow_inf_nan=False,
    )

    content_type: ContentType


class ChatMessageFieldsModel(BaseModel):
    """A base to hang a Config object on for message field models."""

    model_config = ConfigDict(
        extra="allow",
        title="ChatMessageFieldsModel",
        use_enum_values=True,
        cache_strings="keys",
        hide_input_in_errors=True,
        allow_inf_nan=False,
    )


class ChatMetadataModel(BaseModel):
    """Base for all chat metadata models, we allow extra fields because we don't know what the future holds and metadata can be added at any time."""

    model_config = ConfigDict(
        extra="allow",
        title="ChatMetadataModel",
        cache_strings="keys",
        hide_input_in_errors=True,
        allow_inf_nan=False,
    )



# <a name="metadata_start"></a>

# ====================== _SHARED_ MESSAGE METADATA MODELS ======================
"""`Message` metadata models that are used in multiple content types."""


class MessageMetadataModel(ChatMetadataModel):
    """Base for all `Message` metadata models."""

    content_type: ContentType = Field(
        ...,
        exclude=True,
        description="Content type added by our wrapper so we can discriminate the content type.",
    )

    message_type: str | None = Field(
        None,
        description="Type of message -- I think -- I didn't observe a value in my data. Assuming it's a string. Submit an issue if you actually see a value.",
    )
    slug: str | None = Field(
        None,
        description="The slug of the model used to generate the message.",
        examples=["gpt-3.5-turbo", "gpt-4o"],
        alias="model_slug",
    )
    timestamp_: Literal["absolute"] | None = Field(
        None,
        description="*Type* of timestamp; fairly certain the only value is 'absolute'. Submit an issue if you see something different.",
    )
    voice_mode_message: bool | None = Field(
        None,
        description="Attribute indicating if you used voice mode during a conversation session.",
    )
    default_model_slug: str | None = Field(
        None, description="User's selected default model."
    )
    is_complete: bool | None = None
    parent_id: UUID4 | None = None
    request_id: UUID4 | RequestIdType | None = Field(
        None, description="See `RequestIdType`."
    )


class FinishDetails(ChatMetadataModel):
    """Details about message completion. Stop token is the model's token limit if it's hit. These are basically int literals, but they change by model so we'll keep it generic.

    `FinishDetails` is a metadata model that appears in `MessageTextMetadata`, `MessageCodeMetadata`, and `MessageExecutionOutputMetadata` messages.
    """

    item_type: str | Literal["stop"] | None = Field(None, alias="type")
    stop_tokens: list[PositiveInt] | None = None


# Models for citation metadata
class CitationMetadataExtra(ChatMetadataModel):
    """Additional citation metadata giving the index of the cited message."""

    cited_message_idx: PositiveInt
    search_result_idx: PositiveInt | Literal["none"] | None = None
    evidence_text: str | Literal["source"] | None = None


class CitationMetadata(ChatMetadataModel):
    """Metadata associated with a citation."""

    pub_date: PastDatetime | None = None
    text: str
    title: str | None = None
    url: AnyHttpUrl | str | None = None
    extra: CitationMetadataExtra | None = None
    identification: str | None = Field(None, alias="id")
    source: str | None = None
    item_type: str | None = Field(None, alias="type")
    name: str | None = None


class Citations(ChatMetadataModel):
    """Citations for a message. Present for research-oriented bots and in any web results. `Citations` may be present in `MessageTextMetadata` and `MessageCodeMetadata` messages."""

    start_ix: Annotated[PositiveInt, Field(ge=0)]
    end_ix: Annotated[PositiveInt, Field(lt=100000)]
    citation_format_type: Literal["tether_og", "tether_markdown"] | None = None
    metadata: CitationMetadata | None = None
    invalid_reason: str | None = None


class Attachments(ChatMetadataModel):
    """Attachments to a `Message`. Present in `MessageTextMetadata` and `MessageMultimodalTextMetadata` messages."""

    identification: UUID4 | Annotated[str, Field(pattern=r"file-[A-Za-z0-9]{24}")] = (
        Field(..., alias="id")
    )
    name: str = Field(..., description="File name of the attachment.")

    fileTokenSize: int | None = None
    height: int | None = None
    mime_type: str | None = Field(None, alias="mimeType")
    url: AnyHttpUrl | None = None
    content_type: str | None = None
    size: PositiveInt | None = None

    _mime_types: ClassVar[dict[str, str]] = {
        **mimetypes.types_map,
        **mimetypes.common_types,
    } | {
        ".toml": TOML_MIME,
        ".yaml": YAML_MIME,
        ".yml": YAML_MIME,
        ".md": MARKDOWN_MIME,
        ".mk": MARKDOWN_MIME,
    }

    @field_validator("mime_type")
    @classmethod
    def validate_mime(cls, value: str) -> str:
        """Validates a value as a mimetype."""
        mime_map = defaultdict(list)
        for ext, mime in cls._mime_types.items():
            mime_map[mime].append(ext)
        if value not in mime_map and value != "":
            raise ValidationError(f"Invalid mime type: {value}")
        return value

    @property
    def extension(self) -> str:
        """Returns the extension of the attachment file name."""
        if self.mimeType:
            for ext, mime in self._mime_types.items():
                if mime == self.mimeType:
                    return ext
        return ""


# <a name="message_text_metadata"></a>

# ========================= MessageTextMetadata MODELS =========================
"""Metadata models exclusive to `MessageTextContent`/`MessageTextMetadata` messages."""


# JitPluginData models - only present in MessageTextMetadata when plugins are triggered
class JitFromServerBodyActionsTarget(ChatMetadataModel):
    """ID and hash if applicable for a JIT server response to a plugin execution."""

    target_message_id: UUID4
    operation_hash: str | None = None


class JitFromServerBodyActions(ChatMetadataModel):
    """Actions for a JIT server response to a plugin execution."""

    item_type: Literal["allow", "always_allow", "deny"] = Field(..., alias="type")

    allow: JitFromServerBodyActionsTarget | None = None
    always_allow: JitFromServerBodyActionsTarget | None = None
    deny: JitFromServerBodyActionsTarget | None = None
    name: Literal["allow", "decline"] | None = None


class JitFromServerBodyParams(ChatMetadataModel):
    """Parameters for JIT server response to a plugin execution. If the plugin is a search plugin, the URL will be present. The query is the search query."""

    query: str | None = None
    url: AnyHttpUrl | None = None
    year_end: Annotated[PositiveInt, Field(gt=1900, le=2030)] | None = None
    year_start: Annotated[PositiveInt, Field(gt=1900, le=2030)] | None = None


class JitFromServerBody(ChatMetadataModel):
    """Information about the body of a JIT server response to a plugin execution."""

    actions: list[JitFromServerBodyActions]
    domain: Hostname = Field(
        ...,
        description="domain of the plugin server that was triggered.",
        examples=[
            "actions.sider.ai",
        ],
    )
    is_consequential: bool
    method: Literal["get", "post"]
    privacy_policy: AnyHttpUrl

    params: JitFromServerBodyParams | None = None
    path: str | None = Field(None, description="Relative path to file.")


class JitFromServer(ChatMetadataModel):
    """JIT server response to a plugin execution."""

    item_type: Literal["preview"] | str = Field(..., alias="type")
    body: JitFromServerBody | None = None


class JitFromClientTargetData(ChatMetadataModel):
    """Information about the execution of a plugin."""

    item_type: Literal["allow", "always_allow", "deny"] = Field(..., alias="type")
    operation_hash: Annotated[str, Field(pattern=r"^[a-f0-9]*$")] | None = None
    metadata: Any | None = None


class JitFromClientTarget(ChatMetadataModel):
    """target id and data for a plugin execution."""

    data: JitFromClientTargetData
    target_message_id: UUID4


class JitFromClient(ChatMetadataModel):
    """Information about a user action that triggered a Just-In-Time plugin."""

    user_action: JitFromClientTarget


class JitPluginData(ChatMetadataModel):
    """Data for a Just-In-Time plugin. Appears in messages where any external (openai or third party) plugin was triggered by a user action. This can be code execution, internet search, database query, etc."""

    from_client: JitFromClient | None = None
    from_server: JitFromServer | None = None


class UserContextMessageData(ChatMetadataModel):
    """This is user-provided context and instructions that are sent with the message."""

    about_user_message: str | None = Field(
        None, description="User-provided information about the user."
    )
    about_model_message: str | None = Field(
        None, description="User-provided instructions and preferences to the model."
    )


class MessageTextMetadata(MessageMetadataModel):
    """Metadata for a message containing text. This is the most common type of message."""

    content_type: Literal["text"] = "text"
    attachments: list[Attachments] | None = None
    citations: list[Citations] | None = None
    finish_details: FinishDetails | None = None
    gizmo_id: str | None = None
    invoked_plugin: dict[str, Any] | None = None
    is_user_system_message: bool | None = None
    is_visually_hidden_from_conversation: bool | None = None
    jit_plugin_data: JitPluginData | None = Field(
        None,
        description="Not always present, but only appears in `MessageTextMetadata`.",
    )
    message_source: str | None = Field(
        None,
        description="Not always present, and I haven't observed a value for this field; assuming it's a string.",
    )
    pad: Annotated[str, Field(pattern=r"^A*$")] | None = None
    rebase_system_message: bool | None = None
    user_context_message_data: UserContextMessageData | None = None


# <a name="message_code_metadata"></a>

# ========================= MessageCodeMetadata MODELS =========================
"""`MessageCodeMetadata` models are used in `MessageCodeContent` type `Message` objects.`MessageCodeContent` shares some models with `MessageTextContent`. See [shared metadata models](#metadata_start)."""


class MessageCodeMetadata(MessageMetadataModel):
    """Metadata for a message containing code."""

    content_type: Literal["code"] = "code"
    slug: str = Field(..., alias="model_slug")
    citations: list[Citations] | None = None
    finish_details: FinishDetails | None = None
    gizmo_id: str | None = None


# <a name="tether_metadata"></a>

# ================== MessageTetherBrowsingDisplayMetadata ==================
# ======================= MessageTetherQuoteMetadata  =======================


# Citation metadata models specific to tether models, for the `_cite_metadata` attribute.
class CiteMetadataCitationFormat(ChatMetadataModel):
    """Metadata about the citation format. This is a literal value. If regex is present, it matches footnotes in the citation format."""

    name: Literal["tether_og"]
    regex: Pattern | None = None  # regex matches footnotes in the citation format


class InsideCiteMetadataList(ChatMetadataModel):
    """Information about individual citations in a message."""

    text: str
    title: str | None = None
    url: AnyHttpUrl | Annotated[str, Field(eq="")] | None = None
    pub_date: PastDatetime | None = None

    name: str | None = Field(None, description="name of file")
    source: str | None = None


class CiteMetadata(ChatMetadataModel):
    """Metadata about the citations in a message. Metadata metadata metadata... like inception, except cooler."""

    citation_format: CiteMetadataCitationFormat
    metadata_list: list[InsideCiteMetadataList]
    original_query: str | None


class MessageTetherBrowsingDisplayMetadata(MessageMetadataModel):
    """Metadata for a message containing tether browsing display information."""

    content_type: Literal["tether_browsing_display"] = "tether_browsing_display"
    args: list[AnyUrl | str | int | float]
    command: str
    message_type: str | None = None
    slug: str = Field(..., alias="model_slug")
    status: Literal["failed", "finished", "running"] | str
    cite_metadata: CiteMetadata | None = Field(
        None,
        alias="_cite_metadata",
        description="Metadata specific to tether citations.",
    )


class MessageTetherQuoteMetadata(MessageMetadataModel):
    """Metadata for a message containing a tether quote (e.g. a search result)."""

    content_type: Literal["tether_quote"] = "tether_quote"
    command: str
    message_type: str | None
    slug: str = Field(..., alias="model_slug")
    cite_metadata: CiteMetadata | None = Field(
        None,
        alias="_cite_metadata",
        description="Metadata specific to tether citations.",
    )
    args: list[AnyUrl | str | int | float] | None = None
    is_visually_hidden_from_conversation: bool | None = None
    status: Literal["failed", "finished"] | None = None


# <a name="message_execution_output_metadata"></a>

# ======================= MessageExecutionOutput MODELS =======================
"""Metadata models for `MessageExecutionOutputContent` type `Message` objects. These result from Jupyter notebook sessions with Code Interpreter models."""


# AggregateResult models - only present in `MessageExecutionOutputMetadata`/
# `MessageExecutionOutputContent` messages.
class AggregateResultMessage(ChatMetadataModel):
    """Information about a message from a Jupyter notebook execution."""

    message_type: Literal["stream", "image", "timeout_interrupt"]
    sender: Literal["server", "client"]
    time: PastDatetime

    image_payload: Any | None = None  # not observed with a value
    image_url: AnyUrl | None = None
    stream_name: Literal["stdout", "stderr"] | None = None
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
    execution_state: Literal["busy", "idle"] | None = None
    name: Literal["stdout", "stderr"] | None = None
    text: str | None = None
    traceback: list[str] | None = None


class AggregateResultJupyterMessageParentHeader(ChatMetadataModel):
    """Header metadata for a Jupyter notebook execution."""

    msg_id: Annotated[str, Field(pattern=r"^[a-z0-9-]{33}_\d_\d$")] | None = (
        None  # uuid4 with a _ and two integers
    )
    version: PositiveFloat | None = None


class AggregateResultJupyterMessage(ChatMetadataModel):
    """Message from a Jupyter notebook execution."""

    msg_type: Literal[
        "status",
        "execute_input",
        "execute_result",
        "stream",
        "error",
        "@timeout",
        "display_data",
    ]
    parent_header: AggregateResultJupyterMessageParentHeader | None = None
    content: AggregateResultJupyterMessageContent | None = None
    timeout: NonNegativeFloat | None = None


class InKernelException(ChatMetadataModel):
    """Information about an exception that occurred within a Jupyter kernel."""

    args_dict: dict[Hashable, Any] | None = None
    args: list[str] | None
    name: str = Field(..., description="Name of the exception")
    notes: list[str] | None
    traceback: list[str]


class AggregateResult(ChatMetadataModel):
    """Represents the results from a Jupyter notebook execution from the server."""

    code: str = Field(..., description="relative path to the file")
    end_time: PastDatetime | None
    final_expression_output: str | None = Field(
        ..., description="relative path to the file"
    )
    jupyter_messages: list[AggregateResultJupyterMessage]
    messages: list[AggregateResultMessage]
    run_id: UUID4 | Annotated[str, Field(eq="")]
    start_time: PastDatetime
    status: Literal["cancelled", "success", "failed_with_in_kernel_exception"] | str
    update_time: PastDatetime

    in_kernel_exception: InKernelException | None = None
    system_exception: str | None = None
    timeout_triggered: PastDatetime | None = None


class MessageExecutionOutputMetadata(MessageMetadataModel):
    """Metadata for a message containing code execution output."""

    content_type: Literal["execution_output"] = "execution_output"
    aggregate_result: AggregateResult
    is_complete: bool
    slug: str = Field(..., alias="model_slug")
    parent_id: UUID4
    finish_details: FinishDetails | None = None


# <a name="message_system_error_metadata"></a>

# ===================== MessageSystemErrorMetadata MODELS =====================


class MessageSystemErrorMetadata(MessageMetadataModel):
    """Metadata for a system error message. This is a message that contains an error message, usually from a tool like browsing."""

    content_type: Literal["system_error"] = "system_error"
    slug: str = Field(..., alias="model_slug")
    command: str | None = None
    status: Literal["failed", "finished"] | None = None


# <a name="message_multimodal_text_metadata"></a>

# ==================== MessageMultimodalTextMetadata MODELS ====================
"""Metadata models for multimodal text messages. These are messages that contain both text and images, such as DALL-E messages or otherwise use multimodal capabilities."""


class MessageMultimodalTextMetadata(MessageMetadataModel):
    """Metadata for a multimodal text message. This is a message that contains both text and images, such as a DALL-E message."""

    content_type: Literal["multimodal_text"] = "multimodal_text"
    attachments: list[Attachments] | None = Field(
        None,
        description="File attachments associated with the message. Can be either uploaded by user or generated by the model.",
    )
    slug: str | None = Field(None, alias="model_slug")


# ================================ END METADATA ================================

# <a name="message_multimodal_text_content"></a>

# ================== IMAGE CONTENT - MessageMultimodalTextCon ==================


# Dalle and image metadata models are exclusive to `MessageMultimodalTextContent` messages.
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

    dalle: DalleMetadata | None
    gizmo: Any | None
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


class MessageMultimodalTextContent(MessageContentModel):
    """The content of a multi-modal text Message. Includes any DALL-E images, but also can include text in conversations about images or using image recognition."""

    content_type: Literal["multimodal_text"] = "multimodal_text"
    parts: list[str | MessageContentImagePart]


# <a name="message_content"></a>

# ======================== OTHER MESSAGE CONTEXT MODELS ========================


class MessageTextContent(MessageContentModel):
    """The content of a text-based Message (e.g. a chat)."""

    content_type: Literal["text"] = "text"
    parts: list[str]


class MessageCodeContent(MessageContentModel):
    """The content of a code-based Message."""

    content_type: Literal["code"] = "code"
    language: Literal["json", "unknown"] | str = Field(
        ...,
        description="Language of the code block; common values are 'json' and 'unknown'",
    )
    text: str
    response_format_name: str | None = Field(
        None, description="Presumably name of the response format, often None/null."
    )


class MessageTetherBrowsingDisplayContent(MessageContentModel):
    """The content of a tether browsing display Message (search results the model fetches)."""

    content_type: Literal["tether_browsing_display"] = "tether_browsing_display"
    result: str = Field(
        ..., description="Search result text with references to tether quotes."
    )
    summary: str | None = Field(None, description="Summary of the search result.")
    assets: list | None = Field(
        None,
        description="List of assets associated with the search result. Often empty or None.",
    )
    tether_id: str | None = Field(
        None,
        description="Presumably the tether ID of the search result, often None/null.",
    )


class MessageTetherQuoteContent(MessageContentModel):
    """Content of a tether quote Message. Tether quotes are references to external sources from search results."""

    content_type: Literal["tether_quote"] = "tether_quote"
    url: HttpUrl | Annotated[str, Field(pattern=r"file-[A-Za-z0-9]{24}")] = Field(
        ..., description="URL of the tether quote or a file pointer."
    )
    domain: Hostname = Field(
        ...,
        description="Domain of the site that the quote is copied from.",
        examples=["wikipedia.org", "learning.microsoft.com"],
    )
    text: str = Field(..., description="Text of message accompanying quote.")
    title: str
    tether_id: str | None = Field(
        None, description="Presumably tether ID of the tether quote. Often None/null."
    )


class MessageExecutionOutputContent(MessageContentModel):
    """The content of an execution output Message."""

    content_type: Literal["execution_output"] = "execution_output"
    text: str = Field(..., description="Code execution output from the model.")


class MessageSystemErrorContent(MessageContentModel):
    """The content of a system error Message. Appear to be related to failed attempts to handle search results."""

    content_type: Literal["system_error"] = "system_error"
    name: str = Field(
        ...,
        description="Name of the system error.",
        examples=[
            "AssertionError",
            "AceConnectionException",
            "Exception",
            "tool_error",
        ],
    )
    text: str = Field(..., description="Text of the system error")


# =============================== MESSAGE MODELS ===============================

# <a name="message_author"></a>


class MessageAuthor(ChatMessageFieldsModel):
    """The author of a message."""

    metadata: dict[str, Any] | None
    name: str | Literal["browser", "python"] | None
    role: AuthorRole

    @property
    def role_name(self) -> Literal["ChatGPT", "me", "system", "tool"]:
        """Return the role name of the author."""
        return self.role.ui_name


# <a name="message"></a>


class Message(BaseModel):
    """A single message (e.g. user to Assistant or Assistant to user) within a `ChatSession`."""

    author: MessageAuthor = Field(..., description="The author of the message")
    content: MessageContent
    create_time: PastDatetime | None
    end_turn: bool | None
    identification: UUID4 = Field(..., alias="id")
    metadata: MessageMetadata
    recipient: str | Literal["all", "browser", "python"]
    status: str
    update_time: PastDatetime | None
    weight: NonNegativeFloat = Field(..., ge=0)

    def __init__(self, *args, **kwargs) -> None:
        kwargs["metadata"]["content_type"] = kwargs.get("content", {}).get(
            "content_type", ""
        )
        super().__init__(*args, **kwargs)
        self._child_sessions: list[ChatSession] = []
        self._parent_session: ChatSession | None = None

    @property
    def content_type(self) -> str:  # MCS-MIT
        """Get the model type of content in the message."""
        return self.content.content_type

    @property
    def metadata_type(self) -> str:
        """Get the model type of metadata associated with the message."""
        return self.metadata.__class__.__name__

    @property
    def text(self) -> str:  # type: ignore[attr-defined]
        """Get the text content of the message."""
        if self.content.parts:
            return str(self.content.parts[0])
        if self.content.text:
            return self.code_block(self.content.text, self.content.language or "python")
        if self.content.result:
            return self.content.result
        raise ValueError(f"No valid content found in message: {self.id}")

    @staticmethod
    def close_code_blocks(text: str) -> str:  # MCS-MIT
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
    def code_block(text: str, lang: str = "python") -> str:  # MCS-MIT
        """Wrap the given string in a code block."""
        return f"```{lang}\n{text}\n```"


# <a name="chat_session"></a>

# ============================= ChatSession [NODE] =============================


class ChatSession(ChatModel):
    """Class representing a single chat session in a conversation. OpenAI and convoviz call this a 'node', but I think 'session' is more intuitive. It's not a completely precise term, but it's close enough. [I do understand they are referring to tree nodes as a reference to the underlying data structure, but each essentially represents a chat session, so I'm calling it a session]."""

    identification: UUID4 = Field(..., alias="id")
    children: list[UUID4] = Field(default_factory=list)
    message: Message | None = None
    parent: UUID4 | None = None

    _child_sessions: list[Self] | None = []
    _parent_session: Self | None = None

    def add_child(self, session: Self) -> None:  # MCS-MIT
        """Add a child to the session."""
        self._child_sessions.append(session)
        session._parent_session = self

    @classmethod
    def mapping(cls, mapping: dict[str, Self]) -> dict[str, Self]:  # MCS-MIT
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


# <a name="conversation"></a>

# ================================ Conversation ================================


class Conversation(ChatModel):
    """A Conversation is one or more ChatSessions (or nodes) and may be over a wide span of time if the user revisited a conversation."""

    moderation_results: list[Any] | None = []

    conversation_id: UUID4
    conversation_template_id: str | None
    create_time: PastDatetime
    current_node: UUID4
    gizmo_id: str | None
    identification: UUID4 = Field(..., alias="id")
    is_archived: bool
    mapping: dict[UUID4, ChatSession]
    plugin_ids: list[Annotated[str, Field(pattern=r"plugin-[a-f0-9-]{36}")]] | None
    safe_urls: list[str] | list[None]
    title: str
    update_time: PastDatetime

    _added_time: datetime = PrivateAttr(default=datetime.now(UTC))

    @property  # MCS-MIT
    def session_mapping(self) -> dict[str, ChatSession]:
        """Return a dictionary of connected ChatSession objects, based on the mapping."""
        return ChatSession.mapping(self.mapping)

    @property  # MCS-MIT
    def _all_message_sessions(self) -> list[ChatSession]:
        """List of all sessions that have a message, including all branches."""
        return [session for session in self.session_mapping.values() if session.message]

    def _author_sessions(  # MCS-MIT
        self, *authors: Literal["assistant", "user", "system", "tool"]
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
    def leaf_count(self) -> int:  # MCS-MIT
        """Return the number of leaves in the conversation."""
        return sum(
            not session.children_sessions for session in self._all_message_sessions
        )

    @property
    def url(self) -> str:  # MCS-MIT
        """Chat URL."""
        return f"https://chat.openai.com/c/{self.conversation_id}"

    @property
    def content_types(self) -> list[str]:  # MCS-MIT
        """List of all content types in the conversation (all branches)."""
        return list({
            session.message.content.content_type
            for session in self._all_message_sessions
            if session.message
        })

    def message_count(  # MCS-MIT
        self, *authors: Literal["assistant", "user", "system", "tool"]
    ) -> int:
        """Return the number of 'user' and 'assistant' messages (all branches)."""
        if not authors:
            authors = ("user",)
        return len(self._author_sessions(*authors))

    @property
    def llm_model(self) -> str | None:
        """ChatGPT model used for the conversation."""
        assistant_sessions: list[ChatSession] = self._author_sessions("assistant")
        if not assistant_sessions:
            return None

        message = assistant_sessions[0].message

        return message.metadata.slug if message else None

    @property
    def plugins(self) -> list[str]:  # MCS-MIT
        """List of all ChatGPT plugins used in the conversation."""
        return list({
            session.message.metadata.invoked_plugin["namespace"]
            for session in self._author_sessions("tool")
            if session.message and session.message.metadata.invoked_plugin
        })

    @property
    def custom_instructions(self) -> dict[str, str]:  # MCS-MIT
        """Return custom instructions used for the conversation."""
        system_sessions = self._author_sessions("system")
        if len(system_sessions) < 2:
            return {}

        context_message = system_sessions[1].message
        if context_message and context_message.metadata.is_user_system_message:  # type: ignore[attr-defined]
            return context_message.metadata.user_context_message_data or {}

        return {}

    def timestamps(  # MCS-MIT
        self, *authors: Literal["assistant", "user", "system", "tool"]
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
    _idx: int = 0

    @classmethod
    def from_json(cls, file_path: Path) -> Self:
        """Create a ConversationHistoryJson object from a json file."""
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")

        handlers = {
            ".zip": lambda: cls(
                json_obj=Json(
                    cls.read_zip_file(ZipFile(file_path), "conversations.json")
                )
            ),
            ".json": lambda: cls(json_obj=file_path.read_bytes()),
        }

        try:
            return handlers[file_path.suffix]()
        except KeyError as e:
            raise ValueError(
                f"Unsupported file type: {file_path.suffix}. We're expecting conversations.json, or a zip file containing conversations.json."
            ) from e

    @staticmethod
    def read_zip_file(zip_file: ZipFile, file_name: str) -> str:
        """Read a file from a zip file."""
        try:
            with zip_file.open(file_name) as json_file:
                return json_file.read()
        except KeyError as e:
            raise KeyError(
                f"File {file_name} not found in zip file {str(zip_file)}."
            ) from e

    def __len__(self) -> int:
        """Return the number of conversations in the history."""
        return len(self.json_obj)

    def __getitem__(self, idx: int) -> Conversation:
        """Return the conversation at the given index."""
        return self.json_obj[idx]

    def __next__(self) -> Conversation:
        """Return the next conversation in the history."""
        if self._idx >= len(self):
            raise StopIteration
        conversation = self.__getitem__(self._idx)
        self._idx += 1
        return conversation

    def __iter__(self) -> Self:
        """Return an iterator over the conversations in the history."""
        return self

    def dump_json(
        self,
        *,
        indent: int | None = None,
        include: IncEx = None,
        exclude: IncEx = None,
        context: Any | None = None,
        by_alias: bool = True,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool | Literal["none", "warn", "error"] = True,
        serialize_as_any: bool = False,
    ) -> str:
        """Dump the ConversationHistoryJson object to a json string. We switch the default value of `by_alias` to `True`. The rest of the arguments are the same as the BaseModel `model_dump_json` method."""
        return self.model_dump_json(
            self.json_obj,
            indent=indent,
            include=include,
            exclude=exclude,
            context=context,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            round_trip=round_trip,
            warnings=warnings,
            serialize_as_any=serialize_as_any,
        )


def validate_json(path: Path) -> None:
    """Validate conversations.json file. We ca"""
    try:
        return ConversationHistoryJson.from_json(path)
    except ValidationError as e:
        from pprint import pprint

        errors = e.errors()
        raise ValidationError(
            f"Failed to validate conversations. \n {pprint(errors, indent=2)}"
        ) from e


"""
For # MCS-MIT marked code:
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
"""
