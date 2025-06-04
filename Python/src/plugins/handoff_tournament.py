"""
Tournament Contextual Handoff agent.
This agent handles a fighter's journey through a tournament,
but without a manual explicit call from one agent to the next like in the Sequential example.
This one uses descriptions (like semantic kernel plugin descriptions) to route the user to the next agent
given the appropriate context.
It will also dynamically skip steps if you provide the requisite info for previous steps.
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
# Public plugin class
# ---------------------------------------------------------------------------

class TournamentHandoffPlugin:
    """Kernel plugin exposing *handle_customer_query*."""

    TIMEOUT_SECONDS = 15  # overall cap per request

    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.config = OpenAIConfig.from_env()

    # ---------------------------------------------------------------------
    # MAIN ENTRY POINT
    # ---------------------------------------------------------------------

    @kernel_function(
        name="handle_fighter_query",
        description="Route a fighter user input through registration / bullpen / fightRing / RejectedFromTournament agents and return the first agent response.",
    )
    async def handle_fighter_query(
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

        register_agent = _new_agent(
            "RegistrationAgent",
            "Register incoming fighters for the registration.",
            "Decide whether to answer directly or hand off to a specialist agent. You can handoff to the RejectedFromRegistrationAgent if the user does not provide their full name or just says no, or to the BullpenAgent if they have already registered by providing their full name.",
        )
        rejected_registration_agent = _new_agent(
            "RejectedFromRegistrationAgent",
            "Handles fighters that were rejected from the registration.",
            "Ask the user if they would attend next year or not, or if they want to attend this year by providing their full name.",
            # [OrderRefundPlugin()],
        )
        bullpen_agent = _new_agent(
            "BullpenAgent",
            "Checks if the user can pass gear check.",
            "Check if the user passes gear check. They only pass if they have a jacket, pants, gloves, and gorget.",
            # [OrderStatusPlugin()],
        )
        rejected_bullpen_agent = _new_agent(
            "RejectedFromBullpenAgent",
            "Handles fighters that were rejected from the bullpen.",
            "Ask the user if they would attend next year or not, or if they want to attend this year by saying they have a jacket, gorget, gloves, and pants.",
            # [OrderRefundPlugin()],
        )
        ring_agent = _new_agent(
            "RingAgent",
            "User ends up here to complete tournament.",
            "Announce to the user they have arrived at the Ring to compete in the tournament and ends here.",
            # [OrderReturnPlugin()],
        )

        # 2. Handoff graph -------------------------------------------------
        handoffs = (
            OrchestrationHandoffs()
            .add_many(
                source_agent=register_agent.name,
                target_agents={
                    rejected_registration_agent.name: "Transfer if user does not provide their full name or just says no",
                    bullpen_agent.name: "Transfer if user has already registered by providing their full name.",
                    ring_agent.name: "Transfer if the user has already registered by providing their full name and already passed gear check.",
                },
            )
            .add(source_agent=bullpen_agent.name, target_agent=rejected_bullpen_agent.name, description="Transfer if user does not pass gear check. This occurs if they are missing jacket, gorget, gloves, or pants.")
            .add(source_agent=bullpen_agent.name, target_agent=ring_agent.name, description="Transfer if the bullpen approved the user's gear by ensuring the user has a jacket, gorget, pants, and gloves.")
            .add(source_agent=rejected_registration_agent.name, target_agent=register_agent.name, description="Hand back if user provides their full name.")
            .add(source_agent=rejected_bullpen_agent.name, target_agent=bullpen_agent.name, description="Hand back if user says they have a jacket, gloves, pants, and gorget.")
        )

        # 3. Orchestration object -----------------------------------------
        conversation_log: List[str] = []

        def _agent_callback(msg: ChatMessageContent) -> None:
            conversation_log.append(f"{msg.name}: {msg.content}")

        handoff_orch = HandoffOrchestration(
            members=[register_agent, rejected_registration_agent, bullpen_agent, rejected_bullpen_agent, ring_agent],
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
