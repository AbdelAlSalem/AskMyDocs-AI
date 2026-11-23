from uuid import UUID

from app.core.config import Settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationSummary,
    DocumentSelection,
    SavedChatResponse,
)
from app.rag.answer import AnswerService
from app.services.conversations import ConversationStore
from app.services.documents import DocumentService
from app.services.errors import ConversationError


class ChatService:
    def __init__(
        self, documents: DocumentService, answers: AnswerService, conversations: ConversationStore, settings: Settings,
    ) -> None:
        self.documents = documents
        self.answers = answers
        self.conversations = conversations
        self._api_key = settings.groq_api_key.get_secret_value()

    def _ready_scope(self, document_ids: DocumentSelection) -> DocumentSelection:
        if not document_ids:
            return document_ids
        ready = {record.id for record in self.documents.list_documents() if record.status == "ready"}
        return [document_id for document_id in dict.fromkeys(document_ids) if document_id in ready]

    def create(self, document_ids: DocumentSelection) -> ConversationDetail:
        return self.conversations.create(self._ready_scope(document_ids))

    def get(self, conversation_id: UUID) -> ConversationDetail:
        conversation = self.conversations.get(conversation_id)
        return conversation.model_copy(update={"document_ids": self._ready_scope(conversation.document_ids)})

    def rename(self, conversation_id: UUID, title: str) -> ConversationSummary:
        self._reject_credentials(title)
        return self.conversations.rename(conversation_id, title)

    def answer(self, payload: ChatRequest) -> SavedChatResponse:
        self._reject_credentials(payload.question)
        context = self.conversations.context(payload.conversation_id) if payload.conversation_id is not None else None
        document_ids = payload.document_ids
        if context is not None and "document_ids" not in payload.model_fields_set:
            document_ids = self._ready_scope(context.document_ids)

        grounded_request = payload.model_copy(update={
            "document_ids": document_ids,
            "history": context.history if context is not None else [],
        })
        answer = self._redact_answer(self.answers.answer(grounded_request))
        return self.conversations.save_turn(context, payload.question, answer, document_ids)

    def _reject_credentials(self, text: str) -> None:
        if self._api_key and self._api_key in text:
            raise ConversationError(422, "credential_in_text", "Remove the API key from the conversation text and retry.")

    def _redact_answer(self, answer: ChatResponse) -> ChatResponse:
        if not self._api_key:
            return answer
        return answer.model_copy(update={
            "answer": answer.answer.replace(self._api_key, "[REDACTED]"),
            "sources": [
                source.model_copy(update={
                    "filename": source.filename.replace(self._api_key, "[REDACTED]"),
                    "snippet": source.snippet.replace(self._api_key, "[REDACTED]"),
                })
                for source in answer.sources
            ],
        })
