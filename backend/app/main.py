from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI

from app.core.config import Settings

if TYPE_CHECKING:
    from app.services.documents import DocumentService
    from app.services.groq_client import AnswerGenerator


def create_app(
    settings: Settings | None = None,
    *,
    documents: DocumentService | None = None,
    generator: AnswerGenerator | None = None,
) -> FastAPI:
    config = settings if settings is not None else Settings.load()
    if config.demo_mode:
        from app.core.config import DemoSettings
        from app.demo import create_demo_app

        if settings is None:
            demo_config = DemoSettings.load()
        elif isinstance(config, DemoSettings):
            demo_config = config
        else:
            demo_config = DemoSettings.model_validate(config.model_dump())
        return create_demo_app(demo_config, generator=generator)

    from app.local import create_local_app

    return create_local_app(config, documents=documents, generator=generator)


app = create_app()
