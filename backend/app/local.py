from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import Settings
from app.rag.answer import AnswerService
from app.rag.vector_store import VectorStore
from app.services.chat import ChatService
from app.services.conversations import ConversationStore
from app.services.documents import DocumentService
from app.services.errors import ServiceError
from app.services.groq_client import AnswerGenerator, GroqGenerator


def service_error(_request: Request, error: ServiceError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={"detail": {"code": error.code, "message": error.message}},
    )


def create_local_app(
    settings: Settings | None = None,
    *,
    documents: DocumentService | None = None,
    generator: AnswerGenerator | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        config = settings if settings is not None else Settings.load()
        document_store = documents
        if document_store is None:
            vectors = VectorStore(
                directory=config.chroma_dir,
                model_name=config.embedding_model,
                debug_api_key=config.groq_api_key if config.rag_debug_context else None,
            )
            document_store = DocumentService(
                upload_dir=config.upload_dir,
                sqlite_path=config.sqlite_path,
                vectors=vectors,
                max_upload_bytes=config.max_upload_bytes,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
        answer_generator = generator if generator is not None else GroqGenerator(config)
        application.state.settings = config
        application.state.documents = document_store
        answers = AnswerService(document_store, answer_generator, config)
        application.state.chat = ChatService(document_store, answers, ConversationStore(config.conversations_path), config)
        yield

    application = FastAPI(title="AskMyDocs AI", version="1.0.0", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type"],
    )
    application.add_exception_handler(ServiceError, service_error)
    application.include_router(router)
    return application
