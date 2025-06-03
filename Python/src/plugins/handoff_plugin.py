"""
Customer Support Triage Plugin
==============================
This plugin exposes a single kernel function, **handle_customer_query**, that
spins‑up a mini customer‑support triage system based on *Semantic Kernel*’s
**HandoffOrchestration**.  The orchestration contains four specialised agents:

* **TriageAgent** – first‑line support that decides whether to keep the issue
  or hand it off.
* **RefundAgent** – processes refunds.
* **OrderStatusAgent** – checks shipping status.
* **OrderReturnAgent** – processes product returns.

The function is intended to be called from another chat workflow (for example
`chat.py`).  It receives the user’s first question, runs the orchestration long
enough to obtain the first agent reply, and then returns *that reply* as the
function result so that the outer chat loop can display it to the user.

The entire orchestration is created **inside** the function, so no global state
is leaked.  If you need stateful, multi‑turn conversations simply hold on to
`thread` or move the orchestration construction outside the function the same
way `RenamerPlugin` did in your reference file.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Annotated, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function, KernelArguments
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from semantic_kernel.agents import (
    Agent,
    ChatCompletionAgent,
    HandoffOrchestration,
    OrchestrationHandoffs,
)
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.contents import AuthorRole, ChatMessageContent

# ---------------------------------------------------------------------------
# Logging / env config helpers
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
load_dotenv(override=True)

class OpenAIConfig(BaseModel):
    """Simple helper so we do not hard‑code env variable look‑ups everywhere."""

    deployment_name: str = Field(..., description="Azure OpenAI chat deployment")
    api_key: str = Field(..., description="Azure OpenAI key")
    azure_endpoint: str = Field(..., description="Azure endpoint base URL")
    api_version: str = Field(..., description="Azure OpenAI API version")

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        return cls(
            deployment_name=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )


# ---------------------------------------------------------------------------
# Small helper plugins that the specialised agents will call
# ---------------------------------------------------------------------------

class OrderStatusPlugin:
    """Tiny plugin used by *OrderStatusAgent*."""

    @kernel_function(name="check_order_status", description="Return shipping status for an order ID")
    def check_order_status(self, order_id: str) -> str:  # noqa: D401 – simple
        # In real production this would call an order‑management system
        return f"Order {order_id} is shipped and will arrive in 2‑3 days."


class OrderRefundPlugin:
    """Process refunds (used by *RefundAgent*)."""

    @kernel_function(name="process_refund", description="Refund an order if applicable")
    def process_refund(self, order_id: str, reason: str) -> str:  # noqa: D401 – simple
        logger.info("Processing refund – order=%s reason=%s", order_id, reason)
        return f"Refund for order {order_id} has been processed successfully."


class OrderReturnPlugin:
    """Process returns (used by *OrderReturnAgent*)."""

    @kernel_function(name="process_return", description="Create a return authorisation")
    def process_return(self, order_id: str, reason: str) -> str:  # noqa: D401 – simple
        logger.info("Processing return – order=%s reason=%s", order_id, reason)
        return f"Return for order {order_id} has been processed successfully."


# ---------------------------------------------------------------------------
# Main plugin class – this is what your outer chat layer will import
# ---------------------------------------------------------------------------

class CustomerSupportTriagePlugin:
    """Expose *handle_customer_query* as a kernel function for outer chat logic."""

    def __init__(self, kernel: Kernel):
        self.kernel = kernel  # Not used yet, but allows the caller to pass the host Kernel
        # Keep an OpenAI config around in case you need raw client access later
        self.config = OpenAIConfig.from_env()

    # ---------------------------------------------------------------------
    # Public kernel function – feel free to rename or expose more functions
    # ---------------------------------------------------------------------

    @kernel_function(
        name="handle_customer_query",
        description="Route a customer question through triage / refund / status / return agents and return the first agent response.",
    )
    async def handle_customer_query(
        self,
        query: Annotated[str, "The customer's initial message"],
    ) -> Annotated[str, "First reply from the appropriate support agent."]:
        """High‑level entry point that the outer chat loop will call."""
        logger.info("Handling customer query: %s", query)
        # -----------------------------------------------------------------
        # 1. Build the four agents.  We use AzureChatCompletion for each of
        #    them, but you can swap *OpenAIChatCompletion* for local / public
        #    endpoints.
        # -----------------------------------------------------------------
        support_agent = ChatCompletionAgent(
            name="TriageAgent",
            description="A customer support agent that triages issues.",
            instructions="You are first‑line customer support.  Decide whether you can answer the question directly or should hand off to a specialist agent.",
            service=AzureChatCompletion(),
        )

        refund_agent = ChatCompletionAgent(
            name="RefundAgent",
            description="A customer support agent that handles refunds.",
            instructions="Process legitimate refund requests.",
            service=AzureChatCompletion(),
            plugins=[OrderRefundPlugin()],
        )

        order_status_agent = ChatCompletionAgent(
            name="OrderStatusAgent",
            description="A customer support agent that checks shipping status.",
            instructions="Use the OrderStatusPlugin to answer shipping questions.",
            service=AzureChatCompletion(),
            plugins=[OrderStatusPlugin()],
        )

        order_return_agent = ChatCompletionAgent(
            name="OrderReturnAgent",
            description="A customer support agent that handles product returns.",
            instructions="Create RMAs and guide customers through the return process.",
            service=AzureChatCompletion(),
            plugins=[OrderReturnPlugin()],
        )

        # -----------------------------------------------------------------
        # 2. Define the handoff graph so agents know to whom they can defer
        #    a conversation.
        # -----------------------------------------------------------------
        handoffs = (
            OrchestrationHandoffs()
            .add_many(
                source_agent=support_agent.name,
                target_agents={
                    refund_agent.name: "Transfer to this agent if the issue is refund‑related.",
                    order_status_agent.name: "Transfer to this agent if the issue is order status‑related.",
                    order_return_agent.name: "Transfer to this agent if the issue is order return‑related.",
                },
            )
            .add(source_agent=refund_agent.name, target_agent=support_agent.name, description="Hand back if not refund‑related.")
            .add(source_agent=order_status_agent.name, target_agent=support_agent.name, description="Hand back if not status‑related.")
            .add(source_agent=order_return_agent.name, target_agent=support_agent.name, description="Hand back if not return‑related.")
        )

        # -----------------------------------------------------------------
        # 3. Create the orchestration object.
        # -----------------------------------------------------------------
        conversation_log: List[str] = []

        def _agent_callback(msg: ChatMessageContent) -> None:  # noqa: D401
            # Collect but do *not* print – caller can decide what to do.
            conversation_log.append(f"{msg.name}: {msg.content}")

        handoff_orchestration = HandoffOrchestration(
            members=[support_agent, refund_agent, order_status_agent, order_return_agent],
            handoffs=handoffs,
            agent_response_callback=_agent_callback,
            # In this scenario we disable *human_response_function* because the
            # outer chat framework supplies the user messages.  If you need
            # human‑in‑the‑loop you can wire a callback here exactly like in
            # the original sample.
            human_response_function=None,
        )

        # -----------------------------------------------------------------
        # 4. Start a runtime in‑process, invoke orchestration with the user
        #    message, and wait for the *first* answer.
        # -----------------------------------------------------------------
        runtime = InProcessRuntime()
        runtime.start()

        # The *task* is a free‑form string that kicks off the workflow.  We
        # embed the initial user message so the TriageAgent responds to it.
        orchestration_result = await handoff_orchestration.invoke(
            task=f"Incoming customer message: {query}",
            runtime=runtime,
        )

        # Because we only need the first reply to surface back to the caller
        # we just wait for the very first message collected by the callback.
        # If the list is still empty we fall back to the orchestration summary.
        first_reply: str | None = None
        for _ in range(30):  # 3‑second timeout (100 ms × 30)
            await asyncio.sleep(0.1)
            if conversation_log:
                first_reply = conversation_log[0]
                break

        # Make sure we stop cleanly
        await runtime.stop_when_idle()

        if first_reply:
            return first_reply  # e.g. "TriageAgent: Hello! …"
        else:
            return await orchestration_result.get()  # Summary if no replies caught
