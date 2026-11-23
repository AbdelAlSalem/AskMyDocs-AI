"""Optional resident-memory smoke test in the lightweight demo interpreter."""

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[1]
MIB = 1024 * 1024


def _peak_resident_bytes() -> int:
    if sys.platform == "win32":
        result = subprocess.check_output(
            [
                "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                f"(Get-Process -Id {os.getpid()}).PeakWorkingSet64",
            ],
            text=True,
        )
        return int(result.strip())

    from resource import RUSAGE_SELF, getrusage

    return getrusage(RUSAGE_SELF).ru_maxrss * 1024


async def _probe() -> None:
    import httpx

    sys.path.insert(0, str(BACKEND))

    from app.core.config import DemoSettings
    from app.demo import create_demo_app
    from app.services.groq_client import GeneratedAnswer

    for dependency in ("torch", "chromadb", "sentence_transformers"):
        assert importlib.util.find_spec(dependency) is None, dependency

    class OfflineGenerator:
        def generate(self, *, system_prompt, user_prompt, summary=False):
            passages = json.loads(user_prompt)["retrieved_passages"]
            assert sum(len(passage["text"]) for passage in passages) <= 4500
            first = passages[0]
            return GeneratedAnswer(status="answered", claims=[{
                "text": "The cited passage provides the evidence.",
                "evidence": [{"source_id": first["id"], "quote": first["text"]}],
            }])

    app = create_demo_app(DemoSettings(demo_mode=True), generator=OfflineGenerator())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://demo.test") as client:
        responses = await asyncio.gather(*(
            client.post("/api/chat", json={"question": question, "document_ids": ["project-orbit"]})
            for question in (
                "Summarize the main points.",
                "ما أهم النقاط في المستند؟",
                "Are any dates or deadlines mentioned?",
            )
        ))
        assert all(response.status_code == 200 and response.json()["status"] == "answered" for response in responses)
    print(_peak_resident_bytes())


def test_document_wide_demo_stays_within_resident_memory_target():
    import pytest

    demo_python = os.environ.get("ASKMYDOCS_DEMO_PYTHON")
    if not demo_python or sys.platform not in {"win32", "linux"}:
        pytest.skip("Set ASKMYDOCS_DEMO_PYTHON to an isolated demo interpreter on Windows or Linux.")
    run = subprocess.run(
        [demo_python, str(Path(__file__).resolve())],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert int(run.stdout.strip()) < 512 * MIB


if __name__ == "__main__":
    asyncio.run(_probe())
