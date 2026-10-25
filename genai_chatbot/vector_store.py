"""In-memory Chroma collections, one per browser session and document, with local MiniLM embeddings.

Nothing is written to disk: Chroma's ephemeral client keeps collections in process memory, and
all-MiniLM-L6-v2 runs locally on ONNX Runtime (Chroma downloads the model once into its cache).
"""

from __future__ import annotations

import contextlib
import functools
import weakref
from collections.abc import Callable, Sequence
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

from genai_chatbot.observability import new_id
from genai_chatbot.schemas import Chunk, RetrievedPassage

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PASSAGES_PER_QUERY = 4
ADD_BATCH_SIZE = 32


@functools.cache
def get_client() -> ClientAPI:
    """The process's in-memory Chroma client.

    Ephemeral clients share one store per process, so every session's collections live side by side:
    each index gets a unique name and filters its queries by document.
    """
    return chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))


@functools.cache
def get_embedding_function() -> ONNXMiniLM_L6_V2:
    """all-MiniLM-L6-v2 on ONNX Runtime. Its vectors are unit length, so collections use cosine distance."""
    return ONNXMiniLM_L6_V2()


def warm_up() -> None:
    """Load the embedding model now (downloading it on first run) instead of during the first upload."""
    get_embedding_function()(["warm up"])


def _drop_collection(client: ClientAPI, name: str) -> None:
    with contextlib.suppress(NotFoundError):  # already gone, which is the goal
        client.delete_collection(name)


def _metadata(chunk: Chunk) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "document_id": chunk.document_id,
        "filename": chunk.filename,
        "page_number": chunk.page_number,
        "strategy": chunk.strategy,
    }
    if chunk.section_title:
        metadata["section_title"] = chunk.section_title
    return metadata


class DocumentIndex:
    """The searchable passages of one document in one browser session."""

    def __init__(
        self,
        session_id: str,
        document_id: str,
        *,
        client: ClientAPI | None = None,
        embedding_function: Any = None,
    ) -> None:
        self.session_id = session_id
        self.document_id = document_id
        self.name = f"pdfchat-{session_id}-{document_id[:16]}-{new_id()}"
        self._client = client or get_client()
        self._collection = self._client.create_collection(
            self.name,
            embedding_function=embedding_function or get_embedding_function(),
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"session_id": session_id, "document_id": document_id},
        )
        # The collection lives in process memory, so drop it even if the session forgets this index
        # without calling delete(), as when Streamlit discards a closed browser session.
        self._finalizer = weakref.finalize(self, _drop_collection, self._client, self.name)
        self._finalizer.atexit = False  # the whole store goes away with the process anyway

    @property
    def is_deleted(self) -> bool:
        return not self._finalizer.alive

    def add_chunks(
        self, chunks: Sequence[Chunk], on_progress: Callable[[int, int], None] | None = None
    ) -> int:
        """Embed and store chunks in batches, reporting (done, total) after each batch."""
        if self.is_deleted:
            raise RuntimeError("this index has been deleted")
        if any(chunk.document_id != self.document_id for chunk in chunks):
            raise ValueError("chunks from another document can't be added to this index")
        total = len(chunks)
        for start in range(0, total, ADD_BATCH_SIZE):
            batch = chunks[start : start + ADD_BATCH_SIZE]
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=[_metadata(chunk) for chunk in batch],
            )
            if on_progress is not None:
                on_progress(min(start + ADD_BATCH_SIZE, total), total)
        return total

    def count(self) -> int:
        return 0 if self.is_deleted else self._collection.count()

    def search(self, query: str, k: int = PASSAGES_PER_QUERY) -> list[RetrievedPassage]:
        """The passages closest to the query, best first; at most PASSAGES_PER_QUERY of them."""
        if self.is_deleted or not query.strip():
            return []
        limit = min(k, PASSAGES_PER_QUERY, self._collection.count())
        if limit < 1:
            return []
        result = self._collection.query(
            query_texts=[query],
            n_results=limit,
            where={"document_id": self.document_id},
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedPassage(
                chunk_id=chunk_id,
                filename=metadata["filename"],
                page_number=int(metadata["page_number"]),
                text=text,
                section_title=metadata.get("section_title"),
                score=round(1.0 - distance, 4),
            )
            for chunk_id, text, metadata, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]

    def delete(self) -> None:
        """Drop the collection and its embeddings. Safe to call more than once."""
        self._finalizer()
