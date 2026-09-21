"""Strict Pydantic contracts exchanged by the agent pipeline."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DiagnosticFinding(StrictModel):
    issue_id: str = Field(min_length=1, max_length=128)
    severity: Literal["low", "medium", "high", "critical"]
    description: str = Field(min_length=1, max_length=8_000)
    recommended_root_cause_fix: str = Field(min_length=1, max_length=8_000)


class CQIAgentOutput(StrictModel):
    process_efficiency_score: float = Field(ge=0.0, le=100.0)
    primary_bottleneck: str = Field(min_length=1, max_length=4_000)
    identified_waste_areas: list[str] = Field(default_factory=list, max_length=100)
    diagnostic_findings: list[DiagnosticFinding] = Field(default_factory=list, max_length=100)
    compliance_gap_analysis: str = Field(min_length=1, max_length=8_000)
    regulatory_alignment: bool


class TrendMetric(StrictModel):
    metric_name: str = Field(min_length=1, max_length=256)
    historical_baseline: str = Field(min_length=1, max_length=4_000)
    predicted_future_state: str = Field(min_length=1, max_length=4_000)
    confidence_interval: float = Field(ge=0.0, le=1.0)


class RiskProjection(StrictModel):
    trigger_event: str = Field(min_length=1, max_length=4_000)
    likelihood: Literal["unlikely", "possible", "highly_likely"]
    impact: Literal["minor", "moderate", "severe"]


class PredictiveAnalyticsOutput(StrictModel):
    forecast_period: str = Field(min_length=1, max_length=256)
    projected_operational_demand_shift_percentage: float = Field(ge=-100.0, le=1_000.0)
    forecasted_operational_risk_score: float = Field(ge=0.0, le=100.0)
    trend_analysis: list[TrendMetric] = Field(default_factory=list, max_length=100)
    risk_projections: list[RiskProjection] = Field(default_factory=list, max_length=100)


class QuarterlyMilestone(StrictModel):
    quarter: str = Field(min_length=1, max_length=64)
    objective: str = Field(min_length=1, max_length=4_000)
    target_kpis: list[str] = Field(default_factory=list, max_length=100)


class ContingencyPlan(StrictModel):
    risk_factor: str = Field(min_length=1, max_length=4_000)
    severity: Literal["low", "medium", "high", "critical"]
    contingency_action: str = Field(min_length=1, max_length=4_000)


class StrategyAgentOutput(StrictModel):
    quarterly_milestones: list[QuarterlyMilestone] = Field(default_factory=list, max_length=100)
    risk_mitigation_plan: list[ContingencyPlan] = Field(default_factory=list, max_length=100)
    resource_allocation_recommendations: list[str] = Field(default_factory=list, max_length=100)
    executive_summary: str = Field(min_length=1, max_length=8_000)


class CampaignMessaging(StrictModel):
    headline: str = Field(min_length=1, max_length=1_000)
    value_proposition: str = Field(min_length=1, max_length=4_000)
    call_to_action: str = Field(min_length=1, max_length=1_000)


class MarketingAgentOutput(StrictModel):
    executive_summary_deck_outline: list[str] = Field(default_factory=list, max_length=100)
    campaign_messaging: CampaignMessaging
    client_retention_report_copy: str = Field(min_length=1, max_length=8_000)


class HarmonizerOutput(StrictModel):
    answer: str = Field(min_length=1, max_length=8_000)
    confidence: float = Field(ge=0.0, le=1.0)
    cited_node_ids: list[str] = Field(default_factory=list, max_length=50)


class PayloadMeta(StrictModel):
    transaction_id: str = Field(min_length=1, max_length=256)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    client_industry: str = Field(min_length=1, max_length=256)
    subscription_tier: Literal["Standard", "Subscription", "Enterprise"]

    @field_validator("timestamp")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include timezone information")
        return value


class PayloadContext(StrictModel):
    raw_data_summary: dict[str, Any]
    agent_statuses: dict[str, Literal["pending", "completed", "skipped", "failed"]] = Field(
        default_factory=dict
    )
    cqi_insights: CQIAgentOutput | None = None
    predictive_trends: PredictiveAnalyticsOutput | None = None
    strategy_output: StrategyAgentOutput | None = None
    marketing_output: MarketingAgentOutput | None = None

    @field_validator("raw_data_summary")
    @classmethod
    def validate_json_summary(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as error:
            raise ValueError("raw_data_summary must contain JSON-compatible values") from error
        if len(encoded.encode("utf-8")) > 256_000:
            raise ValueError("raw_data_summary exceeds the 256 KiB input limit")
        return value


class ControlFlags(StrictModel):
    requires_predictive_analysis: bool = True
    requires_strategy_synthesis: bool = True
    requires_marketing_collateral: bool = True


class UniversalAgentPayload(StrictModel):
    meta: PayloadMeta
    context: PayloadContext
    control_flags: ControlFlags = Field(default_factory=ControlFlags)
