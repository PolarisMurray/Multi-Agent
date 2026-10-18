"""只有安装了可选依赖（uv sync --extra milvus）时才会运行；用 Milvus Lite，本地文件即可。"""

import pytest

pytest.importorskip("pymilvus")

from bitguide.rag import HashEmbedder, MilvusVectorStore, RagRetriever, TextSplitter  # noqa: E402


async def test_milvus_lite_round_trip(tmp_path):
    store = MilvusVectorStore(str(tmp_path / "milvus.db"), "test_kb", 256)
    retriever = RagRetriever(HashEmbedder(256), store, top_k=1, threshold=0.0)
    documents = [("kids.md", "孩子哭闹时先命名情绪。"), ("db.md", "数据库索引可以加速查询。")]
    assert await retriever.ingest(documents, TextSplitter()) == 2
    assert await store.count() == 2
    [best] = await retriever.retrieve("孩子哭闹怎么办")
    assert best.startswith("[kids.md]")
    await retriever.close()
