import json

import httpx
import pytest

from bitguide.rag import (
    HashEmbedder,
    InMemoryVectorStore,
    OllamaEmbedder,
    RagRetriever,
    TextSplitter,
    cosine,
    load_documents,
)


def test_splitter_respects_size_overlap_and_order():
    text = "。".join(f"第{i}句话的内容稍微长一点" for i in range(40))
    chunks = TextSplitter(50, 10).split(text, "doc.md")
    assert len(chunks) > 1
    assert all(len(c.text) <= 50 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.text[:5] in previous.text  # 相邻块有重叠
    assert "第39句" in chunks[-1].text  # 没有丢掉结尾


def test_splitter_prefers_sentence_boundaries():
    text = "甲" * 30 + "。" + "乙" * 30
    first = TextSplitter(40, 5).split(text, "s")[0]
    assert first.text.endswith("。")


def test_splitter_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        TextSplitter(10, 10)


async def test_hash_embedder_reflects_lexical_similarity():
    a, b, c = await HashEmbedder(128).embed(["孩子发脾气怎么办", "孩子发脾气了", "数据库索引优化"])
    assert cosine(a, b) > cosine(a, c)
    assert abs(sum(x * x for x in a) - 1) < 1e-9  # 已归一化


def test_load_documents_reads_only_text_files(tmp_path):
    (tmp_path / "a.md").write_text("内容A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("内容B", encoding="utf-8")
    (tmp_path / "c.png").write_bytes(b"\x89PNG")
    (tmp_path / "empty.md").write_text("  ", encoding="utf-8")
    assert load_documents(tmp_path) == [("a.md", "内容A"), ("b.txt", "内容B")]


async def test_retriever_ingests_and_retrieves(tmp_path):
    (tmp_path / "kids.md").write_text("孩子哭闹时先命名情绪，再设定边界。", encoding="utf-8")
    (tmp_path / "db.md").write_text("数据库索引可以加速查询，但会拖慢写入。", encoding="utf-8")
    retriever = RagRetriever(HashEmbedder(), InMemoryVectorStore(), top_k=1, threshold=0.0)
    assert await retriever.ingest(load_documents(tmp_path), TextSplitter(100, 10)) == 2
    [best] = await retriever.retrieve("孩子哭闹怎么办")
    assert best.startswith("[kids.md]")
    await retriever.close()


async def test_threshold_filters_weak_matches():
    retriever = RagRetriever(HashEmbedder(), InMemoryVectorStore(), threshold=0.99)
    await retriever.ingest([("a.md", "完全不相关的内容")], TextSplitter())
    assert await retriever.retrieve("孩子") == []


class BrokenEmbedder:
    async def embed(self, texts):
        raise RuntimeError("embedding service down")


async def test_fail_open_degrades_to_no_context():
    assert await RagRetriever(BrokenEmbedder(), InMemoryVectorStore()).retrieve("q") == []
    strict = RagRetriever(BrokenEmbedder(), InMemoryVectorStore(), fail_open=False)
    with pytest.raises(RuntimeError, match="down"):
        await strict.retrieve("q")


async def test_ollama_embedder_request_and_validation():
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        count = len(requests[-1]["input"])
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]] * count})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    embedder = OllamaEmbedder("http://ollama:11434/", "nomic-embed-text", http_client=http)
    assert await embedder.embed(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]
    assert requests[0] == {"model": "nomic-embed-text", "input": ["a", "b"]}
    assert await embedder.embed([]) == []
    await embedder.close()

    bad = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    with pytest.raises(RuntimeError, match="invalid"):
        await OllamaEmbedder("http://x", "m", http_client=bad).embed(["a"])
