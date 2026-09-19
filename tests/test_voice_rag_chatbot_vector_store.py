from voice_rag_chatbot.vector_store import JSONVectorStore, VectorRecord


def _record(id_, value):
    return VectorRecord(id=id_, embedding=[value, 0.0, 0.0], document=f"doc {id_}", metadata={"chunk_index": 0})


def test_upsert_and_persist_round_trip(tmp_path):
    path = tmp_path / "store.json"
    store = JSONVectorStore(path)
    store.upsert([_record("a", 1.0), _record("b", 0.5)])
    assert len(store) == 2

    reloaded = JSONVectorStore(path)
    assert len(reloaded) == 2


def test_upsert_same_id_does_not_duplicate(tmp_path):
    store = JSONVectorStore(tmp_path / "store.json")
    store.upsert([_record("a", 1.0)])
    store.upsert([_record("a", 1.0)])
    assert len(store) == 1


def test_query_returns_closest_by_cosine_similarity(tmp_path):
    store = JSONVectorStore(tmp_path / "store.json")
    store.upsert([_record("close", 1.0), _record("far", -1.0)])
    scored = store.query_with_scores([1.0, 0.0, 0.0], top_k=1)
    assert len(scored) == 1
    score, record = scored[0]
    assert record.id == "close"
    assert score == 1.0


def test_query_on_empty_store_returns_empty(tmp_path):
    store = JSONVectorStore(tmp_path / "store.json")
    assert store.query_with_scores([1.0, 0.0, 0.0]) == []


def test_write_is_atomic_no_tmp_file_left_behind(tmp_path):
    path = tmp_path / "store.json"
    store = JSONVectorStore(path)
    store.upsert([_record("a", 1.0)])
    assert path.is_file()
    assert not list(tmp_path.glob("*.tmp"))
