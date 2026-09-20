"""Tier-aware orchestration for the multi-agent workflow."""

from __future__ import annotations

import json
import logging
from typing import Protocol, TypeVar

from pydantic import BaseModel

from .client import Gemma4VLLMClient
from .prompts import SYSTEM_PROMPTS
from .schemas import (
    CQIAgentOutput,
    MarketingAgentOutput,
    PredictiveAnalyticsOutput,
    StrategyAgentOutput,
    UniversalAgentPayload,
)

logger = logging.getLogger(__name__)
ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class StructuredGenerationClient(Protocol):
    async def generate_structured_response(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[ResponseModel],
        *,
        transaction_id: str | None = None,
    ) -> ResponseModel:
        ...


class SwarmOrchestrator:
    """Run the dependent agent stages for a single validated client payload."""

    def __init__(self, gemma_client: StructuredGenerationClient | Gemma4VLLMClient) -> None:
        self.client = gemma_client

    async def route_task(self, payload: UniversalAgentPayload) -> UniversalAgentPayload:
        result = payload.model_copy(deep=True)
        result.context.agent_statuses = {
            "cqi_agent": "pending",
            "predictive_analytics_agent": "pending",
            "strategy_agent": "pending",
            "marketing_agent": "pending",
        }
        statuses = result.context.agent_statuses
        transaction_id = result.meta.transaction_id

        try:
            result.context.cqi_insights = await self._generate(
                "cqi_agent",
                self._json_context(result),
                CQIAgentOutput,
                transaction_id=transaction_id,
            )
            statuses["cqi_agent"] = "completed"
        except Exception:
            statuses["cqi_agent"] = "failed"
            raise

        is_subscription = result.meta.subscription_tier in {"Subscription", "Enterprise"}
        if not is_subscription or not result.control_flags.requires_predictive_analysis:
            statuses["predictive_analytics_agent"] = "skipped"
        else:
            try:
                result.context.predictive_trends = await self._generate(
                    "predictive_analytics_agent",
                    self._json_context(result),
                    PredictiveAnalyticsOutput,
                    transaction_id=transaction_id,
                )
                statuses["predictive_analytics_agent"] = "completed"
            except Exception:
                statuses["predictive_analytics_agent"] = "failed"
                raise

        is_enterprise = result.meta.subscription_tier == "Enterprise"
        has_forecast = result.context.predictive_trends is not None
        if not is_enterprise or not result.control_flags.requires_strategy_synthesis or not has_forecast:
            statuses["strategy_agent"] = "skipped"
        else:
            try:
                result.context.strategy_output = await self._generate(
                    "strategy_agent",
                    self._json_context(result),
                    StrategyAgentOutput,
                    transaction_id=transaction_id,
                )
                statuses["strategy_agent"] = "completed"
            except Exception:
                statuses["strategy_agent"] = "failed"
                raise

        has_strategy = result.context.strategy_output is not None
        if not is_enterprise or not result.control_flags.requires_marketing_collateral or not has_strategy:
            statuses["marketing_agent"] = "skipped"
        else:
            try:
                result.context.marketing_output = await self._generate(
                    "marketing_agent",
                    self._json_context(result),
                    MarketingAgentOutput,
                    transaction_id=transaction_id,
                )
                statuses["marketing_agent"] = "completed"
            except Exception:
                statuses["marketing_agent"] = "failed"
                raise

        logger.info(
            "agent workflow completed transaction_id=%s tier=%s statuses=%s",
            transaction_id,
            result.meta.subscription_tier,
            statuses,
        )
        return result

    async def _generate(
        self,
        agent_name: str,
        user_prompt: str,
        response_schema: type[ResponseModel],
        *,
        transaction_id: str,
    ) -> ResponseModel:
        return await self.client.generate_structured_response(
            system_prompt=SYSTEM_PROMPTS[agent_name],
            user_prompt=user_prompt,
            response_schema=response_schema,
            transaction_id=transaction_id,
        )

    @staticmethod
    def _json_context(payload: UniversalAgentPayload) -> str:
        return json.dumps(
            {
                "industry": payload.meta.client_industry,
                "subscription_tier": payload.meta.subscription_tier,
                "raw_data_summary": payload.context.raw_data_summary,
                "cqi_insights": payload.context.cqi_insights.model_dump(mode="json")
                if payload.context.cqi_insights
                else None,
                "predictive_trends": payload.context.predictive_trends.model_dump(mode="json")
                if payload.context.predictive_trends
                else None,
                "strategy_output": payload.context.strategy_output.model_dump(mode="json")
                if payload.context.strategy_output
                else None,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
