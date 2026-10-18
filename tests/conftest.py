"""Shared fixtures. Tests run fully offline: hash embedder + fake Groq client (no paid calls)."""
from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import BASE_DIR, Settings
from app.generation.groq_client import GroqClient
from app.services import Services

SAMPLE_DIR = BASE_DIR / "data" / "sample_documents"


class FakeCompletions:
    """Mimics ``groq.Groq().chat.completions`` with a scripted queue of replies/exceptions."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0) if self.replies else self.replies_default
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(model=kw["model"], choices=[SimpleNamespace(message=SimpleNamespace(content=r), finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=321, completion_tokens=45, total_tokens=366))

    replies_default = '{"answerable": false, "answer": "unknown", "citations": []}'


def make_fake_client(replies):
    comp = FakeCompletions(replies)
    return SimpleNamespace(chat=SimpleNamespace(completions=comp)), comp


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(groq_api_key="test-key-not-real", embedding_backend="hash", chroma_persist_directory=str(tmp_path / "chroma"),
                    chroma_collection_name="test_chunks", min_relevance_score=0.15, groq_max_retries=2,
                    allowed_ingest_roots="data", evaluation_output_dir=str(tmp_path / "results"))


@pytest.fixture
def make_services(settings):
    created = []

    def _make(replies=None, **overrides):
        s = dataclasses.replace(settings, **overrides) if overrides else settings
        client, comp = make_fake_client(replies or [])
        llm = GroqClient(s, client=client, sleep=lambda _: None)
        svc = Services(s, llm=llm)
        svc.fake_calls = comp
        created.append(svc)
        return svc
    yield _make
    for s in created:
        s.close()


@pytest.fixture
def services(make_services):
    return make_services()


@pytest.fixture
def ingested(services):
    services.pipeline.ingest_path(SAMPLE_DIR)
    return services


@pytest.fixture
def corpus_copy(tmp_path) -> Path:
    """A private writable copy of the sample corpus under tmp (for modify/delete tests)."""
    dst = tmp_path / "corpus"
    shutil.copytree(SAMPLE_DIR, dst)
    return dst


def answer_json(answerable=True, answer="Employees get 25 days [C1].", citations=("C1",)):
    return json.dumps({"answerable": answerable, "answer": answer, "citations": list(citations)})
