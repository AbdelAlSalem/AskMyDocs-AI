from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


DocumentId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
DocumentSelection = Annotated[list[DocumentId], Field(max_length=100)] | None
ChatStatus = Literal["answered", "insufficient_context"]
ChatRole = Literal["user", "assistant"]
MAX_HISTORY_MESSAGES = 8
MAX_HISTORY_CONTENT = 4000
MAX_CONVERSATION_TITLE = 80


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    groq_configured: bool


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    page_count: int
    chunk_count: int
    size_bytes: int
    status: Literal["processing", "ready", "deleting"]
    created_at: datetime = Field(validation_alias="uploaded_at")


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


class UploadError(BaseModel):
    filename: str
    code: str
    message: str


class UploadResponse(DocumentListResponse):
    errors: list[UploadError]


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: ChatRole
    content: str = Field(min_length=1, max_length=MAX_HISTORY_CONTENT)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2000)
    conversation_id: UUID | None = None
    document_ids: DocumentSelection = None
    history: list[HistoryMessage] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)


class SourceReference(BaseModel):
    id: int
    document_id: str
    filename: str
    page_number: int
    chunk_id: str
    chunk_ids: list[str]
    snippet: str


class ChatResponse(BaseModel):
    status: ChatStatus
    answer: str
    sources: list[SourceReference]


class ConversationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_ids: DocumentSelection = None


class ConversationRename(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=MAX_CONVERSATION_TITLE)


class ConversationSummary(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationMessage(BaseModel):
    id: UUID
    conversation_id: UUID
    role: ChatRole
    content: str
    created_at: datetime
    source_references: list[SourceReference] | None = None


class ConversationDetail(ConversationSummary):
    document_ids: DocumentSelection
    messages: list[ConversationMessage]


class ConversationList(BaseModel):
    conversations: list[ConversationSummary]


class SavedChatResponse(ChatResponse):
    conversation_id: UUID
    conversation: ConversationSummary
    messages: list[ConversationMessage]
