import os
import asyncio

from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies.termination.termination_strategy import TerminationStrategy
from semantic_kernel.agents.strategies.selection.kernel_function_selection_strategy import (
    KernelFunctionSelectionStrategy,
)
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.kernel import Kernel

ROOKIE_INSTRUCTIONS = "You are a rookie at japanese and occassionally make mistakes. Don't respond to the user input. \
        You create a brand new sentence in English that you make up, then try to translate it to Japanese with a couple mistakes for your teacher to fix."

TRANSLATOR_INSTRUCTIONS = "You are a translator who translates between languages. After translating, append the string &&&."

TERMINATOR_INSTRUCTIONS = "After the Translator translates a string, return the string &&&."

from plugins.translators_plugin import TranslatorPlugins


class ApprovalTerminationStrategy(TerminationStrategy):
    """A strategy for determining when an agent should terminate."""

    async def should_agent_terminate(self, agent, history):
        """Check if the agent should terminate."""
        return any("&&&" in message.content for message in history)


async def run_multi_agent(input: str):
    """Run the multi-agent system. Setup the kernel, agents, and group chat with a termination strategy."""
    service_id_rookie ="rookie"
    service_id_teacher = "translator"
    # Define the Kernel
    kernel = Kernel()
    # kernel.add_service(AzureChatCompletion(service_id=service_id_rookie))
    # settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id_rookie)
    kernel.add_service(AzureChatCompletion(service_id=service_id_teacher))
    settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id_teacher)
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    kernel.add_plugin(TranslatorPlugins(kernel), plugin_name="TranslatorPlugins")

    # rookie_agent = ChatCompletionAgent(
    #     id=service_id_rookie,
    #     kernel=kernel,
    #     name="Rookie",
    #     instructions= ROOKIE_INSTRUCTIONS,
    # )
    translator_agent = ChatCompletionAgent(
        id=service_id_teacher,
        kernel=kernel,
        name="Translator",
        instructions=TRANSLATOR_INSTRUCTIONS,
    )

    chat = AgentGroupChat(
        agents=[ translator_agent],
        termination_strategy=ApprovalTerminationStrategy(agents=[translator_agent], maximum_iterations=6),
    )
    await chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=input))

    # Collect responses
    responses = []
    async for response in chat.invoke():
        responses.append({"role": response.role.value, "message": response.content})

    return responses