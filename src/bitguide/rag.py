"""RAG：切块 → 向量化 → 入库 → 检索。

每一层都是 Protocol + 可替换实现：
    Embedder    : HashEmbedder（离线）| OllamaEmbedder（真实语义向量）
    VectorStore : InMemoryVectorStore（零依赖）| MilvusVectorStore（生产向量库）
RagRetriever 只依赖这两个协议，满足 contracts.Retriever，可以直接塞给 Agent。
"""

from __future__ import annotations

import hashlib
import math
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx


@dataclass(slots=True, frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    index: int


def load_documents(path: str | Path) -> list[tuple[str, str]]:
    """读取目录下所有 .md / .txt，返回 (来源, 全文)。"""
    root = Path(path)
    files = [root] if root.is_file() else sorted(root.rglob("*"))
    documents = []
    for file in files:
        if file.is_file() and file.suffix.lower() in {".md", ".markdown", ".txt"}:
            text = file.read_text(encoding="utf-8").strip()
            if text:
                documents.append((file.name, text))
    return documents


class TextSplitter:
    """按最大长度切块，尽量在段落 / 句号处断开；相邻块重叠 overlap 个字符。"""

    SEPARATORS = ("\n\n", "。", "！", "？", "\n", ". ", " ")

    def __init__(self, max_segment_size: int = 300, overlap: int = 50):
        if max_segment_size < 1 or not 0 <= overlap < max_segment_size:
            raise ValueError("splitter requires 0 <= overlap < max_segment_size")
        self.max_segment_size = max_segment_size
        self.overlap = overlap

    def split(self, text: str, source: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        start = 0
        while start < len(text):
            end = min(start + self.max_segment_size, len(text))
            if end < len(text):
                # 从窗口末尾往前找最近的自然边界；太靠前（不到半个窗口）就放弃，硬切
                boundary = max(text.rfind(sep, start, end) for sep in self.SEPARATORS)
                if boundary > start + self.max_segment_size // 2:
                    end = boundary + 1
            piece = text[start:end].strip()
            if piece:
                digest = hashlib.sha256(f"{source}:{len(chunks)}:{piece}".encode()).hexdigest()
                chunks.append(Chunk(digest[:24], piece, source, len(chunks)))
            if end >= len(text):
                break
            # 重叠让跨块的句子不会丢上下文；max 保证游标一定前进，不会死循环
            start = max(start + 1, end - self.overlap)
        return chunks


class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashEmbedder:
    """哈希技巧（hashing trick）：把字符二元组散列到固定维度并归一化。

    它只反映“字面重合度”，不懂语义，作用是让 RAG 流程在完全离线时也能跑通、能测试。
    """

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        normalized = "".join(text.casefold().split())
        grams = [normalized[i : i + 2] for i in range(len(normalized) - 1)] or [normalized]
        for gram in grams:
            # crc32 在不同进程之间结果稳定；内置 hash() 每次启动都会变
            vector[zlib.crc32(gram.encode()) % self.dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class OllamaEmbedder:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 30,
        http_client: httpx.AsyncClient | None = None,
    ):
        self._http = http_client or httpx.AsyncClient(timeout=timeout)
        self._url = base_url.rstrip("/") + "/api/embed"
        self._model = model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._http.post(
            self._url, json={"model": self._model, "input": list(texts)}
        )
        response.raise_for_status()
        embeddings = response.json().get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama returned an invalid embedding response")
        return embeddings

    async def close(self) -> None:
        await self._http.aclose()


class VectorStore(Protocol):
    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None: ...

    async def search(self, vector: list[float], top_k: int) -> list[tuple[float, Chunk]]: ...

    async def count(self) -> int: ...

    async def close(self) -> None: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class InMemoryVectorStore:
    """暴力遍历 + 余弦相似度。几百个块完全够用；数据量大了再换 Milvus。"""

    def __init__(self) -> None:
        self._rows: dict[str, tuple[Chunk, list[float]]] = {}

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._rows[chunk.id] = (chunk, list(vector))

    async def search(self, vector: list[float], top_k: int) -> list[tuple[float, Chunk]]:
        scored = [(cosine(vector, stored), chunk) for chunk, stored in self._rows.values()]
        return sorted(scored, key=lambda pair: pair[0], reverse=True)[:top_k]

    async def count(self) -> int:
        return len(self._rows)

    async def close(self) -> None:
        self._rows.clear()


class MilvusVectorStore:
    """Milvus 实现。需要 `uv sync --extra milvus`；uri 可以是本地文件（Milvus Lite）。

    注意：pymilvus 是同步 API，这里直接调用会短暂阻塞事件循环。
    TODO(改写): 用 asyncio.to_thread 包一层，或者换成 pymilvus 的 AsyncMilvusClient。
    """

    def __init__(self, uri: str, collection: str, dimension: int):
        # 延迟导入：不用 Milvus 就不需要装（uv sync --extra milvus）
        from pymilvus import DataType, MilvusClient  # pyright: ignore[reportMissingImports]

        self._client = MilvusClient(uri=uri)
        self._collection = collection
        self._dimension = dimension
        if not self._client.has_collection(collection_name=collection):
            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
            schema.add_field("text", DataType.VARCHAR, max_length=8192)
            schema.add_field("source", DataType.VARCHAR, max_length=1024)
            schema.add_field("chunk_index", DataType.INT64)
            index = self._client.prepare_index_params()
            index.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
            self._client.create_collection(collection, schema=schema, index_params=index)

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        rows = [
            {
                "id": chunk.id,
                "vector": vector,
                "text": chunk.text,
                "source": chunk.source,
                "chunk_index": chunk.index,
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if rows:
            self._client.upsert(self._collection, data=rows)

    async def search(self, vector: list[float], top_k: int) -> list[tuple[float, Chunk]]:
        result = self._client.search(
            self._collection,
            data=[vector],
            limit=top_k,
            output_fields=["text", "source", "chunk_index"],
            search_params={"metric_type": "COSINE"},
        )
        hits: list[tuple[float, Chunk]] = []
        for hit in result[0] if result else []:
            entity: dict[str, Any] = hit.get("entity", {})
            chunk = Chunk(
                str(hit.get("id")),
                str(entity.get("text", "")),
                str(entity.get("source", "")),
                int(entity.get("chunk_index", 0)),
            )
            hits.append((float(hit.get("distance", 0.0)), chunk))
        return hits

    async def count(self) -> int:
        stats = self._client.get_collection_stats(self._collection)
        return int(stats.get("row_count", 0))

    async def close(self) -> None:
        self._client.close()


class RagRetriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        *,
        top_k: int = 3,
        threshold: float = 0.1,
        fail_open: bool = True,
    ):
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._threshold = threshold
        self._fail_open = fail_open

    async def ingest(self, documents: Sequence[tuple[str, str]], splitter: TextSplitter) -> int:
        chunks = [chunk for source, text in documents for chunk in splitter.split(text, source)]
        if not chunks:
            return 0
        # TODO(改写): 文档多时分批 embed，避免一次请求过大
        vectors = await self._embedder.embed([chunk.text for chunk in chunks])
        await self._store.upsert(chunks, vectors)
        return len(chunks)

    async def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        try:
            [vector] = await self._embedder.embed([query])
            hits = await self._store.search(vector, top_k or self._top_k)
        except Exception:
            # fail-open：知识库挂了只是“没资料可参考”，Agent 仍然可以用模型和工具回答
            if self._fail_open:
                return []
            raise
        # 压成一行：Agent 会把每段资料作为 System Prompt 里的一个列表项
        return [
            f"[{chunk.source}] {' '.join(chunk.text.split())}"
            for score, chunk in hits
            if score >= self._threshold
        ]

    async def close(self) -> None:
        await self._store.close()
        close = getattr(self._embedder, "close", None)
        if close is not None:
            await close()
