"""
Translation Pipeline Plugin
===========================
Exposes **translate_en_to_fr** as a Semantic‑Kernel kernel‑function. It
translates an English sentence through the chain:

```
English -> Hindi -> Japanese -> Spanish -> Portuguese -> French
```

For each hop we spin up a dedicated `ChatCompletionAgent` and stream every
intermediate translation back to the caller.  If any step fails (timeout or
Azure content‑filter) we inject a `System:` message and skip the remaining
hops while still returning whatever translations succeeded so far.

**Note:** All arrow symbols use the ASCII sequence `->` (no Unicode arrows) so
this file can be processed by tools that reject non‑ASCII characters.
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
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai.exceptions.content_filter_ai_exception import (
    ContentFilterAIException,
)

logger = logging.getLogger(__name__)
load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Azure‑OpenAI config helper
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


# ---------------------------------------------------------------------------
# Main plugin class
# ---------------------------------------------------------------------------

class TranslationPipelinePlugin:
    """Kernel plugin exposing the English->...->French pipeline."""

    TIMEOUT_SECONDS = 15

    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.config = OpenAIConfig.from_env()

    # Utility – quick agent factory -------------------------------------
    def _mk_agent(self, name: str, instructions: str) -> ChatCompletionAgent:
        return ChatCompletionAgent(
            name=name,
            description=instructions.split(".")[0],
            instructions=instructions,
            service=AzureChatCompletion(),
        )

    # ------------------------------------------------------------------
    # Kernel function entry‑point
    # ------------------------------------------------------------------

    @kernel_function(
        name="translate_en_to_fr",
        description="Translate English -> Hindi -> Japanese -> Spanish -> Portuguese -> French and return all intermediate translations.",
    )
    async def translate_en_to_fr(
        self,
        text: Annotated[str, "English text to translate"],
    ) -> Annotated[str, "Multi‑line string with translations for every hop."]:
        # Build agents ---------------------------------------------------
        en_hi = self._mk_agent(
            "English_to_Hindi",
            "Translate strictly from English to Hindi. Return only the Hindi text.",
        )
        hi_ja = self._mk_agent(
            "Hindi_to_Japanese",
            "Translate strictly from Hindi to Japanese. Return only the Japanese text.",
        )
        ja_es = self._mk_agent(
            "Japanese_to_Spanish",
            "Translate strictly from Japanese to Spanish. Return only the Spanish text.",
        )
        es_pt = self._mk_agent(
            "Spanish_to_Portuguese",
            "Translate strictly from Spanish to Portuguese. Return only the Portuguese text.",
        )
        pt_fr = self._mk_agent(
            "Portuguese_to_French",
            "Translate strictly from Portuguese to French. Return only the French text.",
        )

        # Safe‑translate helper that **always returns str** -------------
        async def _safe_translate(agent: ChatCompletionAgent, src: str) -> str:
            try:
                response = await asyncio.wait_for(
                    agent.get_response(messages=src),
                    timeout=self.TIMEOUT_SECONDS,
                )
                content_obj = getattr(response, "content", response)
                if isinstance(content_obj, str):
                    text_out = content_obj
                else:
                    text_out = getattr(content_obj, "content", str(content_obj))
                return text_out.strip()
            except asyncio.TimeoutError:
                msg = f"System: {agent.name} timed out after {self.TIMEOUT_SECONDS}s."
                logger.error(msg)
                return msg
            except ContentFilterAIException as cf_ex:
                msg = (
                    f"System: Azure content filter blocked a response from {agent.name}. "
                    "Please adjust your input."
                )
                logger.warning("Content filter hit: %s", cf_ex)
                return msg
            except Exception as ex:
                logger.exception("Unexpected error in %s: %s", agent.name, ex)
                return f"System: {agent.name} encountered an error."

        # Helper to chain steps -----------------------------------------
        async def _step(prev_text: str, agent: ChatCompletionAgent) -> str:
            if prev_text.startswith("System:"):
                return "System: Skipped due to previous error."
            return await _safe_translate(agent, prev_text)

        # Pipeline execution --------------------------------------------
        conversation: List[str] = []

        hindi = await _safe_translate(en_hi, text)
        conversation.append(f"{en_hi.name}: {hindi}")

        japanese = await _step(hindi, hi_ja)
        conversation.append(f"{hi_ja.name}: {japanese}")

        spanish = await _step(japanese, ja_es)
        conversation.append(f"{ja_es.name}: {spanish}")

        portuguese = await _step(spanish, es_pt)
        conversation.append(f"{es_pt.name}: {portuguese}")

        french = await _step(portuguese, pt_fr)
        conversation.append(f"{pt_fr.name}: {french}")

        return "\n".join(conversation)
