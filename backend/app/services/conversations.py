import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import TypeAdapter

from app.models.schemas import (
    MAX_CONVERSATION_TITLE,
    MAX_HISTORY_CONTENT,
    MAX_HISTORY_MESSAGES,
    ChatResponse,
    ConversationDetail,
    ConversationMessage,
    ConversationSummary,
    DocumentSelection,
    HistoryMessage,
    SavedChatResponse,
    SourceReference,
)
from app.services.errors import ConversationError


_SOURCES_JSON = TypeAdapter(list[SourceReference])
_NEW_TITLE = "New conversation"


@dataclass(frozen=True, slots=True)
class ConversationContext:
    id: UUID
    revision: int
    document_ids: DocumentSelection
    history: list[HistoryMessage]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _initial_title(question: str) -> str:
    text = " ".join(question.split())
    if len(text) <= MAX_CONVERSATION_TITLE:
        return text
    return text[: MAX_CONVERSATION_TITLE - 1].rstrip() + "…"


def _summary(row: sqlite3.Row) -> ConversationSummary:
    return ConversationSummary(id=row["id"], title=row["title"], created_at=row["created_at"], updated_at=row["updated_at"])


def _message(row: sqlite3.Row) -> ConversationMessage:
    sources = row["source_references"]
    return ConversationMessage(
        id=row["id"],
        conversation_id=row["conversation_id"],
        role=row["role"],
        content=row["content"],
        created_at=row["created_at"],
        source_references=_SOURCES_JSON.validate_json(sources) if sources is not None else None,
    )


class ConversationStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection(write=True) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    document_ids TEXT NOT NULL,
                    title_edited INTEGER NOT NULL DEFAULT 0,
                    revision INTEGER NOT NULL DEFAULT 0
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    sequence INTEGER PRIMARY KEY,
                    id TEXT NOT NULL UNIQUE,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    source_references TEXT
                )
            """)
            connection.execute("CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages(conversation_id, sequence)")

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        try:
            with closing(sqlite3.connect(self.path)) as connection, connection:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                yield connection
        except sqlite3.Error:
            raise ConversationError(503, "conversation_storage_unavailable", "Conversation storage is unavailable.") from None

    @staticmethod
    def _get(connection: sqlite3.Connection, conversation_id: UUID) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM conversations WHERE id = ?", (str(conversation_id),)).fetchone()
        if row is None:
            raise ConversationError(404, "conversation_not_found", "This conversation could not be found.")
        return row

    @staticmethod
    def _insert(connection: sqlite3.Connection, document_ids: DocumentSelection, timestamp: str) -> UUID:
        conversation_id = uuid4()
        connection.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at, document_ids) VALUES (?, ?, ?, ?, ?)",
            (str(conversation_id), _NEW_TITLE, timestamp, timestamp, json.dumps(document_ids)),
        )
        return conversation_id

    def list(self) -> list[ConversationSummary]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM conversations ORDER BY updated_at DESC, created_at DESC, id")
            return [_summary(row) for row in rows]

    def create(self, document_ids: DocumentSelection) -> ConversationDetail:
        with self._connection(write=True) as connection:
            conversation_id = self._insert(connection, document_ids, _now())
            summary = _summary(self._get(connection, conversation_id))
            return ConversationDetail(**summary.model_dump(), document_ids=document_ids, messages=[])

    def get(self, conversation_id: UUID) -> ConversationDetail:
        with self._connection() as connection:
            row = self._get(connection, conversation_id)
            messages = connection.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY sequence",
                (str(conversation_id),),
            )
            return ConversationDetail(
                **_summary(row).model_dump(),
                document_ids=json.loads(row["document_ids"]),
                messages=[_message(message) for message in messages],
            )

    def rename(self, conversation_id: UUID, title: str) -> ConversationSummary:
        with self._connection(write=True) as connection:
            connection.execute(
                "UPDATE conversations SET title = ?, title_edited = 1, updated_at = ? WHERE id = ?",
                (title, _now(), str(conversation_id)),
            )
            return _summary(self._get(connection, conversation_id))

    def delete(self, conversation_id: UUID) -> None:
        with self._connection(write=True) as connection:
            self._get(connection, conversation_id)
            connection.execute("DELETE FROM conversations WHERE id = ?", (str(conversation_id),))

    def context(self, conversation_id: UUID) -> ConversationContext:
        with self._connection() as connection:
            row = self._get(connection, conversation_id)
            messages = connection.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY sequence DESC LIMIT ?",
                (str(conversation_id), MAX_HISTORY_MESSAGES),
            ).fetchall()
            return ConversationContext(
                id=conversation_id,
                revision=row["revision"],
                document_ids=json.loads(row["document_ids"]),
                history=[
                    HistoryMessage(role=message["role"], content=message["content"][:MAX_HISTORY_CONTENT])
                    for message in reversed(messages)
                ],
            )

    def save_turn(
        self, context: ConversationContext | None, question: str, answer: ChatResponse, document_ids: DocumentSelection,
    ) -> SavedChatResponse:
        with self._connection(write=True) as connection:
            timestamp = _now()
            conversation_id = self._insert(connection, document_ids, timestamp) if context is None else context.id
            row = self._get(connection, conversation_id)
            revision = row["revision"]
            if context is not None and revision != context.revision:
                raise ConversationError(409, "conversation_changed", "The conversation changed. Reload it and retry.")

            title = _initial_title(question) if revision == 0 and not row["title_edited"] else row["title"]
            messages = [
                ConversationMessage(id=uuid4(), conversation_id=conversation_id, role="user", content=question, created_at=timestamp),
                ConversationMessage(
                    id=uuid4(), conversation_id=conversation_id, role="assistant", content=answer.answer,
                    created_at=timestamp, source_references=answer.sources,
                ),
            ]
            connection.executemany(
                """
                INSERT INTO messages (id, conversation_id, role, content, created_at, source_references)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(message.id), str(conversation_id), message.role, message.content, timestamp,
                        _SOURCES_JSON.dump_json(message.source_references).decode("utf-8")
                        if message.source_references is not None else None,
                    )
                    for message in messages
                ],
            )
            connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ?, document_ids = ?, revision = ? WHERE id = ?",
                (title, timestamp, json.dumps(document_ids), revision + 1, str(conversation_id)),
            )
            return SavedChatResponse(
                **answer.model_dump(),
                conversation_id=conversation_id,
                conversation=_summary(self._get(connection, conversation_id)),
                messages=messages,
            )
