from quantum_atomic_rag.memory import KnowledgeStore


def test_store_ingests_deduplicates_and_retrieves(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path / "memory.sqlite3"))
    first = store.ingest("Quantum memory stores durable context", source="paper")
    duplicate = store.ingest("Quantum memory stores durable context", source="other")

    assert first.node_id == duplicate.node_id
    assert store.count() == 1

    result = store.search("durable context")
    assert len(result.retrieved_nodes) == 1
    assert result.retrieved_nodes[0].momentum > 0
    assert store.graph()["nodes"][0]["node_id"] == first.node_id
