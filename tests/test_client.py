import httpx
import pytest

from quantum_atomic_rag.client import Gemma4VLLMClient
from quantum_atomic_rag.config import RuntimeSettings
from quantum_atomic_rag.schemas import CQIAgentOutput


def cqi_response() -> dict:
    return {
        "process_efficiency_score": 75,
        "primary_bottleneck": "Manual review",
        "identified_waste_areas": ["Duplicate entry"],
        "diagnostic_findings": [],
        "compliance_gap_analysis": "None found",
        "regulatory_alignment": True,
    }


@pytest.mark.asyncio
async def test_client_enforces_json_schema_request_format() -> None:
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(request.read().decode() and __import__("json").loads(request.read()))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": __import__("json").dumps(cqi_response())}}]},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = Gemma4VLLMClient(
            RuntimeSettings(vllm_endpoint_url="http://test/v1", max_retries=0),
            http_client=http_client,
        )
        result = await client.generate_structured_response("system", "client data", CQIAgentOutput)

    assert result.process_efficiency_score == 75
    assert captured["response_format"]["type"] == "json_schema"
    assert "untrusted client data" in captured["messages"][1]["content"]


@pytest.mark.asyncio
async def test_client_can_enable_gemma_thinking_per_request() -> None:
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(__import__("json").loads(request.read()))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": __import__("json").dumps(cqi_response())}}]},
        )

    settings = RuntimeSettings(
        vllm_endpoint_url="http://test/v1",
        max_retries=0,
        enable_thinking=True,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = Gemma4VLLMClient(settings, http_client=http_client)
        await client.generate_structured_response("system", "client data", CQIAgentOutput)

    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}


@pytest.mark.asyncio
async def test_client_retries_transient_failure() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": __import__("json").dumps(cqi_response())}}]},
        )

    transport = httpx.MockTransport(handler)
    settings = RuntimeSettings(
        vllm_endpoint_url="http://test/v1",
        max_retries=1,
        retry_backoff_seconds=0.001,
    )
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = Gemma4VLLMClient(settings, http_client=http_client)
        await client.generate_structured_response("system", "client data", CQIAgentOutput)

    assert attempts == 2
