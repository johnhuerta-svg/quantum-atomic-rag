import pytest
from pydantic import ValidationError

from quantum_atomic_rag.schemas import PayloadContext, PayloadMeta


def test_payload_meta_uses_timezone_aware_timestamp() -> None:
    metadata = PayloadMeta(transaction_id="tx-1", client_industry="FinTech", subscription_tier="Standard")

    assert metadata.timestamp.tzinfo is not None
    assert metadata.timestamp.utcoffset() is not None


def test_payload_context_rejects_non_json_values() -> None:
    with pytest.raises(ValidationError, match="JSON-compatible"):
        PayloadContext(raw_data_summary={"bad": object()})


def test_payload_context_rejects_oversized_values() -> None:
    with pytest.raises(ValidationError, match="256 KiB"):
        PayloadContext(raw_data_summary={"data": "x" * 256_001})
