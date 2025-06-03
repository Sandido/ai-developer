"""
Customer Support Triage Plugin
==============================
Exposes **handle_customer_query** as a Semantic‑Kernel *kernel function* that
instantiates a mini customer‑support triage orchestration (triage → refund /
status / return agents).  The function is designed to be called from an outer
chat workflow and now includes robust error‑handling and a *hard timeout* so
callers are never left hanging.  Any timeout / safety‑filter event is surfaced
back to the caller both as the function return value **and** as a synthetic
"System:" message injected into the agent callback stream so that a live chat
UI immediately displays the issue.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Annotated, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import (
    ChatCompletionAgent,
    HandoffOrchestration,
    OrchestrationHandoffs,
)
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.contents import ChatMessageContent, AuthorRole

from semantic_kernel.connectors.ai.open_ai.exceptions.content_filter_ai_exception import (
    ContentFilterAIException,
)

logger = logging.getLogger(__name__)
load_dotenv(override=True)

class OpenAIConfig(BaseModel):
    deployment_name: str = Field(...)
    api_key: str = Field(...)
    azure_endpoint: str = Field(...)
    api_version: str = Field(...)

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        return cls(
            deployment_name=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )


# ---------------------------------------------------------------------------
# Helper plugins used by specialised agents
# ---------------------------------------------------------------------------

class OrderStatusPlugin:
    @kernel_function(name="check_order_status", description="Return shipping status for an order ID")
    def check_order_status(self, order_id: str) -> str:
        return f"Order {order_id} is shipped and will arrive in 2‑3 days."


class OrderRefundPlugin:
    @kernel_function(name="process_refund", description="Refund an order if applicable")
    def process_refund(self, order_id: str, reason: str) -> str:
        logger.info("Processing refund – order=%s reason=%s", order_id, reason)
        return f"Refund for order {order_id} has been processed successfully."


class OrderReturnPlugin:
    @kernel_function(name="process_return", description="Create a return authorisation")
    def process_return(self, order_id: str, reason: str) -> str:
        logger.info("Processing return – order=%s reason=%s", order_id, reason)
        return f"Return for order {order_id} has been processed successfully."


# ---------------------------------------------------------------------------
# Public plugin class
# ---------------------------------------------------------------------------

class CustomerSupportTriagePlugin:
    """Kernel plugin exposing *handle_customer_query*."""

    TIMEOUT_SECONDS = 15  # overall cap per request

    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.config = OpenAIConfig.from_env()

    # ---------------------------------------------------------------------
    # MAIN ENTRY POINT
    # ---------------------------------------------------------------------

    @kernel_function(
        name="handle_customer_query",
        description="Route a customer question through triage / refund / status / return agents and return the first agent response.",
    )
    async def handle_customer_query(
        self,
        query: Annotated[str, "The customer's initial message"],
    ) -> Annotated[str, "First agent reply or a System error message."]:
        # 1. Build agents --------------------------------------------------
        def _new_agent(n: str, d: str, i: str, plugins: list | None = None):
            return ChatCompletionAgent(
                name=n,
                description=d,
                instructions=i,
                service=AzureChatCompletion(),
                plugins=plugins or [],
            )

        support_agent = _new_agent(
            "TriageAgent",
            "Triages incoming customer issues.",
            "Decide whether to answer directly or hand off to a specialist agent.",
        )
        refund_agent = _new_agent(
            "RefundAgent",
            "Processes refunds.",
            "Handle legitimate refund requests.",
            [OrderRefundPlugin()],
        )
        order_status_agent = _new_agent(
            "OrderStatusAgent",
            "Checks order shipping status.",
            "Use OrderStatusPlugin to answer shipping queries.",
            [OrderStatusPlugin()],
        )
        order_return_agent = _new_agent(
            "OrderReturnAgent",
            "Processes product returns.",
            "Create RMAs and guide customers through returns.",
            [OrderReturnPlugin()],
        )

        # 2. Handoff graph -------------------------------------------------
        handoffs = (
            OrchestrationHandoffs()
            .add_many(
                source_agent=support_agent.name,
                target_agents={
                    refund_agent.name: "Transfer if refund‑related.",
                    order_status_agent.name: "Transfer if order status‑related.",
                    order_return_agent.name: "Transfer if order return‑related.",
                },
            )
            .add(source_agent=refund_agent.name, target_agent=support_agent.name, description="Hand back if not refund‑related.")
            .add(source_agent=order_status_agent.name, target_agent=support_agent.name, description="Hand back if not status‑related.")
            .add(source_agent=order_return_agent.name, target_agent=support_agent.name, description="Hand back if not return‑related.")
        )

        # 3. Orchestration object -----------------------------------------
        conversation_log: List[str] = []

        def _agent_callback(msg: ChatMessageContent) -> None:
            conversation_log.append(f"{msg.name}: {msg.content}")

        handoff_orch = HandoffOrchestration(
            members=[support_agent, refund_agent, order_status_agent, order_return_agent],
            handoffs=handoffs,
            agent_response_callback=_agent_callback,
            human_response_function=None,
        )

        # 4. Runtime + invoke ---------------------------------------------
        runtime = InProcessRuntime()
        runtime.start()

        # ---- helper to push synthetic system message --------------------
        def _push_system(msg: str) -> None:
            conversation_log.append(f"System: {msg}")

        try:
            orch_result = await handoff_orch.invoke(
                task=f"Incoming customer message: {query}",
                runtime=runtime,
            )
        except ContentFilterAIException as cf_ex:
            err_msg = (
                "Apologies, your request triggered our safety filters. "
                "Please rephrase and avoid disallowed content."
            )
            logger.warning("Prompt blocked: %s", cf_ex)
            _push_system(err_msg)
            await runtime.stop_when_idle()
            return err_msg
        except Exception as ex:
            err_msg = (
                "I'm sorry — I'm having trouble connecting you to support right now. "
                "Please try again in a few minutes."
            )
            logger.exception("Error invoking orchestration: %s", ex)
            _push_system(err_msg)
            await runtime.stop_when_idle()
            return err_msg

        # 5. Wait for reply / summary with timeout ------------------------
        first_reply: str | None = None
        try:
            for _ in range(int(self.TIMEOUT_SECONDS * 10)):
                await asyncio.sleep(0.1)
                if conversation_log:
                    first_reply = conversation_log[0]
                    break

            if not first_reply:
                summary = await asyncio.wait_for(
                    orch_result.get(), timeout=self.TIMEOUT_SECONDS
                )
                return summary
        except asyncio.TimeoutError:
            err_msg = (
                "Support is taking longer than expected. Please try again later or "
                "contact us through another channel."
            )
            logger.error("Orchestration timed out after %s s", self.TIMEOUT_SECONDS)
            _push_system(err_msg)
            return err_msg
        except ContentFilterAIException as cf_ex:
            err_msg = (
                "Apologies, our safety system could not generate a response. "
                "Please adjust your request."
            )
            logger.warning("Response blocked: %s", cf_ex)
            _push_system(err_msg)
            return err_msg
        except Exception as ex:
            err_msg = (
                "I wasn't able to retrieve a response from support. "
                "Please rephrase your question or try again later."
            )
            logger.exception("Error retrieving orchestration result: %s", ex)
            _push_system(err_msg)
            return err_msg
        finally:
            try:
                await runtime.stop_when_idle()
            except Exception as ex:
                logger.exception("Error stopping runtime: %s", ex)

        # 6. Normal path ---------------------------------------------------
        return first_reply if first_reply else (
            "Support team could not provide a response at this time. "
            "Please try again shortly."
        )
