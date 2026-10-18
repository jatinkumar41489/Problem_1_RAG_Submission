"""OPTIONAL live test. Skipped unless RUN_LIVE_GROQ=1 and GROQ_API_KEY are set (makes real, billable-by-quota API calls)."""
import os

import pytest

from app.config import Settings
from app.services import Services
from tests.conftest import SAMPLE_DIR

pytestmark = pytest.mark.skipif(not (os.environ.get("RUN_LIVE_GROQ") == "1" and os.environ.get("GROQ_API_KEY")),
                                reason="set RUN_LIVE_GROQ=1 and GROQ_API_KEY to run live Groq tests")


def test_live_grounded_answer(tmp_path):
    s = Settings.from_env()
    import dataclasses
    s = dataclasses.replace(s, chroma_persist_directory=str(tmp_path / "c"), chroma_collection_name="live_test")
    svc = Services(s)
    svc.pipeline.ingest_path(SAMPLE_DIR)
    r = svc.answers.answer("How many days of paid annual leave do full-time employees receive?")
    svc.close()
    assert r.status == "answered" and "25" in r.answer and r.citations and r.token_usage["available"]


def test_live_unanswerable_question_does_not_hallucinate(tmp_path):
    s = Settings.from_env()
    import dataclasses
    s = dataclasses.replace(s, chroma_persist_directory=str(tmp_path / "c"), chroma_collection_name="live_test2", min_relevance_score=-1.0)
    svc = Services(s)
    svc.pipeline.ingest_path(SAMPLE_DIR)
    r = svc.answers.answer("What is the company policy on bringing pets to the office?")
    svc.close()
    assert r.status in ("insufficient_evidence", "no_relevant_context")
