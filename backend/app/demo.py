from typing import Literal

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import DemoSettings
from app.models.schemas import (
    MAX_HISTORY_MESSAGES,
    ChatRequest,
    ChatResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentSelection,
    HistoryMessage,
)
from app.rag.answer import AnswerService
from app.rag.demo_retrieval import DemoRetriever
from app.services.demo_security import TURNSTILE_MAX_TOKEN_CHARS, DemoAbuseGuard
from app.services.errors import ServiceError
from app.services.groq_client import AnswerGenerator, GroqGenerator


class DemoHealth(BaseModel):
    status: Literal["ok"] = "ok"
    groq_configured: bool
    demo_mode: Literal[True] = True
    turnstile_site_key: str | None = None
    turnstile_action: str | None = None
    max_question_length: int


class DemoChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2000)
    document_ids: DocumentSelection = None
    history: list[HistoryMessage] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)
    turnstile_token: str | None = Field(default=None, max_length=TURNSTILE_MAX_TOKEN_CHARS)


def _bounded_history(history: list[HistoryMessage], config: DemoSettings) -> list[HistoryMessage]:
    remaining = config.demo_max_history_chars
    recent: list[HistoryMessage] = []
    first = max(0, len(history) - config.demo_max_history_messages)
    for message in reversed(history[first:]):
        if not remaining:
            break
        content = message.content[-remaining:]
        remaining -= len(content)
        recent.append(message.model_copy(update={"content": content}))
    return recent[::-1]


def create_demo_app(
    config: DemoSettings,
    *,
    generator: AnswerGenerator | None = None,
    retriever: DemoRetriever | None = None,
) -> FastAPI:
    corpus = retriever if retriever is not None else DemoRetriever(max_context_chars=config.demo_max_context_chars)
    guard = DemoAbuseGuard(config)
    answer_settings = config.model_copy(update={
        "rag_top_k": config.demo_top_k,
        "rag_min_similarity": 0.30,
        "rag_debug_context": False,
    })
    answer_generator = generator if generator is not None else GroqGenerator(config)
    answers = AnswerService(corpus, answer_generator, answer_settings)
    application = FastAPI(
        title="AskMyDocs AI (public demo)",
        max_body_size=config.demo_max_body_bytes,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=config.demo_allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        expose_headers=["Retry-After"],
    )

    def service_error(_request: Request, error: ServiceError) -> JSONResponse:
        headers = {"Cache-Control": "no-store"}
        retry_after = getattr(error, "retry_after", None)
        if retry_after is not None:
            headers["Retry-After"] = str(retry_after)
        return JSONResponse(
            status_code=error.status_code, headers=headers,
            content={"detail": {"code": error.code, "message": error.message}},
        )

    def validation_error(_request: Request, _error: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": {"code": "invalid_request", "message": "Check the question and try again."}},
        )

    application.add_exception_handler(ServiceError, service_error)
    application.add_exception_handler(RequestValidationError, validation_error)

    @application.get("/api/health", response_model=DemoHealth)
    async def health() -> DemoHealth:
        return DemoHealth(
            groq_configured=config.groq_configured,
            turnstile_site_key=config.demo_turnstile_site_key or None,
            turnstile_action=config.demo_turnstile_action if config.turnstile_required else None,
            max_question_length=config.demo_max_question_chars,
        )

    @application.get("/api/documents", response_model=DocumentListResponse)
    async def documents() -> DocumentListResponse:
        return DocumentListResponse(documents=[
            DocumentResponse.model_validate(document) for document in corpus.list_documents()
        ])

    @application.post("/api/chat", response_model=ChatResponse)
    def chat(payload: DemoChatRequest, request: Request, response: Response) -> ChatResponse:
        if len(payload.question) > config.demo_max_question_chars:
            raise ServiceError(
                422, "demo_question_too_long",
                f"Keep your question to {config.demo_max_question_chars} characters or fewer.",
            )
        with guard.authorize(
            request.client.host if request.client else None, payload.turnstile_token,
            render_client_hosts=request.headers.getlist("CF-Connecting-IP"),
        ):
            result = answers.answer(ChatRequest(
                question=payload.question,
                document_ids=payload.document_ids,
                history=_bounded_history(payload.history, config),
            ))
        response.headers["Cache-Control"] = "no-store"
        return result

    @application.post("/api/documents/upload")
    @application.delete("/api/documents/{document_id}")
    async def documents_are_read_only() -> None:
        raise ServiceError(403, "demo_read_only", "The public demo uses read-only sample documents.")

    @application.api_route("/api/conversations", methods=["GET", "POST"])
    @application.api_route("/api/conversations/{conversation_id}", methods=["GET", "PATCH", "DELETE"])
    async def conversation_storage_is_disabled() -> None:
        raise ServiceError(403, "demo_browser_history", "Demo chats stay in this browser.")

    return application
