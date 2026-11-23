import pytest

from app.core.config import Settings


@pytest.fixture
def make_settings(tmp_path):
    def configure(**overrides):
        return Settings(
            upload_dir=tmp_path / "uploads",
            sqlite_path=tmp_path / "documents.sqlite3",
            chroma_dir=tmp_path / "chroma",
            conversations_path=tmp_path / "conversations.sqlite3",
            **overrides,
        )

    return configure
