"""
Translation Pipeline Plugin – Interactive Handoff
================================================
This version turns the pipeline into an **interactive translator** driven by a
Router agent.  Flow:

1. **RouterAgent** – if the user hasn’t specified a language, asks which of the
   supported languages they want (Hindi, Japanese, Spanish, Portuguese, French)
   and ends with `HANDOFF: <TargetAgent>`.
2. **Target translator agents** (e.g. `English_to_Hindi`) – translate the
   user’s English sentence into their language, then ask if the user wants
   another translation.  They finish with `HANDOFF: RouterAgent` (or
   `HANDOFF: END` if the user says “stop”).

The hand‑off graph lets the Router send the conversation to any translator, and
lets every translator hand back to the Router.

The results of this have been tricky. The handoff logic needs to be much more specific to ensure each later agent
is called correctly. 
DELETE.
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
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.connectors.ai.open_ai.exceptions.content_filter_ai_exception import (
    ContentFilterAIException,
)

logger = logging.getLogger(__name__)
load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Optional Azure‑OpenAI config helper
# ---------------------------------------------------------------------------

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


SUPPORTED = {
    "Hindi": "English_to_Hindi",
    "Japanese": "English_to_Japanese",
    "Spanish": "English_to_Spanish",
    "Portuguese": "English_to_Portuguese",
    "French": "English_to_French",
}

# ---------------------------------------------------------------------------
# Main plugin class
# ---------------------------------------------------------------------------

class TranslationPipelinePlugin:
    """Interactive translator via SK hand‑off orchestration."""

    TIMEOUT_SECONDS = 30

    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.config = OpenAIConfig.from_env()

    # Helper to create translator agent ---------------------------------
    def _translator(self, lang: str) -> ChatCompletionAgent:
        name = SUPPORTED[lang]
        remaining_langs = [l for l in SUPPORTED.keys() if l != lang]
        ask_more = (
            "If the user wants another translation, ask them which of the following languages they want next: "
            + ", ".join(remaining_langs)
            + ". If they say 'stop', end with 'HANDOFF: END', else end with 'HANDOFF: RouterAgent'."
        )
        instructions = (
            f"Translate strictly from English to {lang}. Return only the {lang} text.\n"
            + ask_more
        )
        return ChatCompletionAgent(
            name=name,
            description=f"Translates English to {lang}",
            instructions=instructions,
            service=AzureChatCompletion(),
        )

    # Router agent ------------------------------------------------------
    def _router(self) -> ChatCompletionAgent:
        language_list = ", ".join(SUPPORTED.keys())
        instructions = (
            "You are the router. If the user message includes an explicit target language (" + language_list + "), "
            "reply ONLY with 'HANDOFF: <AgentName>' where <AgentName> is the correct translator.\n"
            "If the user says 'stop', reply 'HANDOFF: END'.\n"
            "If no language specified, ask the user: 'Which language would you like? [" + language_list + "]' and end with 'HANDOFF: RouterAgent'."
        )
        return ChatCompletionAgent(
            name="RouterAgent",
            description="Routes requests to language‑specific translators",
            instructions=instructions,
            service=AzureChatCompletion(),
        )

    # ------------------------------------------------------------------
    # Kernel entry‑point
    # ------------------------------------------------------------------

    @kernel_function(name="interactive_translate", description="Interactively translate English text to user‑selected languages.")
    async def interactive_translate(
        self,
        text: Annotated[str, "English text or a command like 'stop'."],
    ) -> Annotated[str, "Conversation log of the interactive translation."]:
        # Build agents --------------------------------------------------
        router = self._router()
        translators = {lang: self._translator(lang) for lang in SUPPORTED}

        # Handoff graph: Router -> any translator, each translator -> Router ---
        handoffs = OrchestrationHandoffs()
        for lang, agent_name in SUPPORTED.items():
            handoffs.add(source_agent=router.name, target_agent=agent_name)
            handoffs.add(source_agent=agent_name, target_agent=router.name)
        # Final END agent (implicit) not needed; orchestration ends when no handoff

        # Collect messages ---------------------------------------------
        convo: List[str] = []

        def _cb(msg: ChatMessageContent):
            convo.append(f"{msg.name}: {msg.content}")

        orch = HandoffOrchestration(
            members=[router, *translators.values()],
            handoffs=handoffs,
            agent_response_callback=_cb,
            human_response_function=lambda: ChatMessageContent.from_user(input("User: ")),  # basic CLI prompt
        )

        runtime = InProcessRuntime()
        runtime.start()

        try:
            res = await orch.invoke(task=text, runtime=runtime)
            await asyncio.wait_for(res.get(), timeout=self.TIMEOUT_SECONDS)
        except Exception as ex:
            convo.append(f"System: Error – {ex}")
        finally:
            try:
                await runtime.stop_when_idle()
            except Exception:
                pass

        return "\n".join(convo)
