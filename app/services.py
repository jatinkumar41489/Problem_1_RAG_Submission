"""Service container: builds encoder, store, pipeline, retriever and answer service from Settings."""
from __future__ import annotations

from app.config import Settings
from app.embeddings.encoder import Encoder, get_encoder
from app.generation.answer_service import AnswerService
from app.generation.groq_client import GroqClient
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.retriever import Retriever
from app.vectorstore.chroma_store import ChromaStore


class Services:
    def __init__(self, settings: Settings, encoder: Encoder | None = None,
                 store: ChromaStore | None = None, llm: GroqClient | None = None) -> None:
        self.settings = settings
        self.encoder = encoder or get_encoder(settings)
        self.store = store or ChromaStore(settings.persist_path, settings.chroma_collection_name,
                                          self.encoder.name, self.encoder.dim)
        self.llm = llm or GroqClient(settings)
        self.pipeline = IngestionPipeline(settings, self.encoder, self.store)
        self.retriever = Retriever(settings, self.encoder, self.store)
        self.answers = AnswerService(settings, self.retriever, self.llm)
        self.last_ingest: dict | None = None

    def close(self) -> None:
        self.store.close()
