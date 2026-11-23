from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Request, Response, UploadFile
from starlette.concurrency import run_in_threadpool

from app.models.schemas import (
    ChatRequest,
    ConversationCreate,
    ConversationDetail,
    ConversationList,
    ConversationRename,
    ConversationSummary,
    DocumentId,
    DocumentListResponse,
    DocumentResponse,
    HealthResponse,
    SavedChatResponse,
    UploadError,
    UploadResponse,
)
from app.services.chat import ChatService
from app.services.documents import DocumentError, DocumentService


router = APIRouter(prefix="/api")


def document_service(request: Request) -> DocumentService:
    return request.app.state.documents


def chat_service(request: Request) -> ChatService:
    return request.app.state.chat


Documents = Annotated[DocumentService, Depends(document_service)]
Chat = Annotated[ChatService, Depends(chat_service)]


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(groq_configured=request.app.state.settings.groq_configured)


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(documents: Documents) -> DocumentListResponse:
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(record) for record in documents.list_documents()]
    )


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(
    files: Annotated[list[UploadFile], File()],
    documents: Documents,
    response: Response,
) -> UploadResponse:
    uploaded = []
    errors = []
    error_statuses = set()
    for file in files:
        filename = file.filename or ""
        try:
            record = await run_in_threadpool(documents.ingest, filename, file.file)
            uploaded.append(DocumentResponse.model_validate(record))
        except DocumentError as error:
            display_name = filename.replace("\\", "/").rsplit("/", 1)[-1][:255]
            errors.append(UploadError(filename=display_name, code=error.code, message=error.message))
            error_statuses.add(error.status_code)
        finally:
            await file.close()

    if errors and not uploaded:
        response.status_code = next(iter(error_statuses)) if len(error_statuses) == 1 else 422
        if any(code >= 500 for code in error_statuses):
            response.status_code = max(error_statuses)
    return UploadResponse(documents=uploaded, errors=errors)


@router.delete("/documents/{document_id}", status_code=204, response_class=Response)
def delete_document(document_id: DocumentId, documents: Documents) -> Response:
    documents.delete(document_id)
    return Response(status_code=204)


@router.get("/conversations", response_model=ConversationList)
def list_conversations(chat: Chat) -> ConversationList:
    return ConversationList(conversations=chat.conversations.list())


@router.post("/conversations", status_code=201, response_model=ConversationDetail)
def create_conversation(chat: Chat, payload: ConversationCreate | None = None) -> ConversationDetail:
    return chat.create(payload.document_ids if payload is not None else None)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: UUID, chat: Chat) -> ConversationDetail:
    return chat.get(conversation_id)


@router.patch("/conversations/{conversation_id}", response_model=ConversationSummary)
def rename_conversation(conversation_id: UUID, payload: ConversationRename, chat: Chat) -> ConversationSummary:
    return chat.rename(conversation_id, payload.title)


@router.delete("/conversations/{conversation_id}", status_code=204, response_class=Response)
def delete_conversation(conversation_id: UUID, chat: Chat) -> Response:
    chat.conversations.delete(conversation_id)
    return Response(status_code=204)


@router.post("/chat", response_model=SavedChatResponse)
def ask_question(payload: ChatRequest, chat: Chat) -> SavedChatResponse:
    return chat.answer(payload)
