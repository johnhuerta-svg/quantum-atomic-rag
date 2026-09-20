import pytest

from quantum_atomic_rag.orchestrator import SwarmOrchestrator
from quantum_atomic_rag.schemas import (
    CQIAgentOutput,
    MarketingAgentOutput,
    PredictiveAnalyticsOutput,
    StrategyAgentOutput,
    UniversalAgentPayload,
)


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate_structured_response(self, system_prompt, user_prompt, response_schema, *, transaction_id=None):
        self.calls.append(response_schema.__name__)
        if response_schema is CQIAgentOutput:
            return CQIAgentOutput(
                process_efficiency_score=70,
                primary_bottleneck="Backlog",
                compliance_gap_analysis="None",
                regulatory_alignment=True,
            )
        if response_schema is PredictiveAnalyticsOutput:
            return PredictiveAnalyticsOutput(
                forecast_period="Q1",
                projected_operational_demand_shift_percentage=5,
                forecasted_operational_risk_score=20,
            )
        if response_schema is StrategyAgentOutput:
            return StrategyAgentOutput(executive_summary="Scale carefully")
        return MarketingAgentOutput(
            campaign_messaging={"headline": "Grow", "value_proposition": "More", "call_to_action": "Start"},
            client_retention_report_copy="Retain clients",
        )


def payload(tier: str) -> UniversalAgentPayload:
    return UniversalAgentPayload(
        meta={"transaction_id": "tx-1", "client_industry": "FinTech", "subscription_tier": tier},
        context={"raw_data_summary": {"backlog": 12}},
    )


@pytest.mark.asyncio
async def test_standard_tier_marks_downstream_agents_skipped() -> None:
    client = FakeClient()
    request = payload("Standard")
    result = await SwarmOrchestrator(client).route_task(request)

    assert client.calls == ["CQIAgentOutput"]
    assert result.context.agent_statuses == {
        "cqi_agent": "completed",
        "predictive_analytics_agent": "skipped",
        "strategy_agent": "skipped",
        "marketing_agent": "skipped",
    }
    assert request.context.cqi_insights is None


@pytest.mark.asyncio
async def test_enterprise_tier_runs_all_stages() -> None:
    client = FakeClient()
    result = await SwarmOrchestrator(client).route_task(payload("Enterprise"))

    assert client.calls == [
        "CQIAgentOutput",
        "PredictiveAnalyticsOutput",
        "StrategyAgentOutput",
        "MarketingAgentOutput",
    ]
    assert all(status == "completed" for status in result.context.agent_statuses.values())
