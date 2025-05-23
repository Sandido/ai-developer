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

from typing import Awaitable, Callable
from semantic_kernel.filters import FilterTypes, FunctionInvocationContext

from semantic_kernel.agents.strategies import DefaultTerminationStrategy, SequentialSelectionStrategy

ROOKIE_INSTRUCTIONS = "Your role in this collaboration is: You are a rookie at japanese and occassionally make mistakes. Don't respond to the user input. \
        You create a brand new sentence in English that you make up, then try to translate it to Japanese with a couple mistakes for your teacher to fix. \
        Do not mention what your mistake was. \
        Prepend your response with the string Rookie Student Agent: "

TRANSLATOR_INSTRUCTIONS = "Your role in this collaboration is: You are a translator who translates between Japanese and English. Correct any translation mistakes. \
                        Prepend your response with the string Translator Agent: "

TERMINATOR_INSTRUCTIONS = "Your role in this collaboration is: After the Translator Agent translates a string, return the string &&&. \
                        Prepend your response with the string Terminator Agent: "
# after testing, the agents don't know which agent just ran inherently. Maybe we can add it to the user messages. 
# Not even then does the terminator agent work. 

from plugins.translators_plugin import TranslatorPlugins


class ApprovalTerminationStrategy(TerminationStrategy):
    """A strategy for determining when an agent should terminate."""

    async def should_agent_terminate(self, agent, history):
        """Check if the agent should terminate."""
        return any("&&&" in message.content for message in history)

async def function_invocation_filter(
    context: FunctionInvocationContext,
    next: Callable[[FunctionInvocationContext], Awaitable[None]],
) -> None:
    # this runs before the function is called
    print(f"  ---> Calling Plugin {context.function.plugin_name}.{context.function.name} with arguments `{context.arguments}`")
    # let's await the function call
    await next(context)
    # this runs after our functions has been called
    print(f"  ---> Plugin response from [{context.function.plugin_name}.{context.function.name} is `{context.result}`")


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
    
    kernel.add_filter(FilterTypes.FUNCTION_INVOCATION, function_invocation_filter)

    # kernel.add_plugin(TranslatorPlugins(kernel), plugin_name="TranslatorPlugins")
    
    
    # original ones I was working with
    # rookie_agent = ChatCompletionAgent(
    #     id=service_id_rookie,
    #     kernel=kernel,
    #     name="Rookie",
    #     instructions= ROOKIE_INSTRUCTIONS,
    # )
    # translator_agent = ChatCompletionAgent(
    #     id=service_id_teacher,
    #     kernel=kernel,
    #     name="Translator",
    #     instructions=TRANSLATOR_INSTRUCTIONS,
    # )

    # chat = AgentGroupChat(
    #     agents=[ translator_agent],
    #     termination_strategy=ApprovalTerminationStrategy(agents=[translator_agent], maximum_iterations=6),
    # )
    

    # Create a writer agent that generates content
    writer = ChatCompletionAgent(
        service=AzureChatCompletion(),
        name="Writer",
        instructions="""You are a creative writer who crafts engaging content.
        Prepend your response with the string Writer Agent:
        Your role in this collaboration:
        1. Generate original content based on the topic provided
        2. Apply creative storytelling techniques to engage readers
        3. Incorporate feedback from the editor and fact-checker to improve your writing
        4. Revise content to address issues raised by other team members
        5. Focus on creating a compelling narrative voice and structure
        
        When responding to feedback:
        - Be open to constructive criticism
        - Explain your creative choices when relevant
        - Incorporate suggestions that improve the content
        
        Always strive to maintain the core message while making the content more engaging and effective. Keep your responses very concise, imaginative and engaging.""",
    )

    # Create an editor agent that improves the writing
    editor = ChatCompletionAgent(
        service=AzureChatCompletion(),
        name="Editor",
        instructions="""You are a meticulous editor who improves content quality and clarity.
        Prepend your response with the string Editor Agent:
        Your role in this collaboration:
        1. Review content for clarity, coherence, and flow
        2. Identify and fix grammatical, structural, or stylistic issues
        3. Suggest improvements to enhance readability and impact
        4. Ensure the content meets its intended purpose and audience needs
        5. Maintain consistent voice and tone throughout
        
        When providing feedback:
        - Be specific about what needs improvement and why
        - Offer constructive suggestions rather than just criticism
        - Consider both micro (sentence-level) and macro (structure) improvements
        - Balance preserving the writer's voice with improving the content
        
        Your goal is to elevate the writing while respecting the writer's intent and style. Keep your responses very concise, clear and straightforward.""",
    )

    # Create a fact-checker agent that ensures accuracy
    fact_checker = ChatCompletionAgent(
        service=AzureChatCompletion(),
        name="FactChecker",
        instructions="""You are a thorough fact-checker who ensures accuracy and credibility.
        Prepend your response with the string Fact Checker Agent:
        Your role in this collaboration:
        1. Verify factual claims in the content
        2. Identify potential inaccuracies or misleading statements
        3. Suggest corrections for any factual errors
        4. Recommend additional context where needed for accuracy
        5. Ensure the content is truthful and well-supported
        
        When providing feedback:
        - Focus on accuracy rather than style or structure
        - Explain why a statement might be problematic
        - Provide correct information to replace inaccuracies
        - Consider potential sources of factual support
        
        Your goal is to ensure the content maintains high standards of accuracy and integrity. Keep your responses very concise, clear and straightforward.""",
    )

    # Create a group chat with all specialized agents
    chat = AgentGroupChat(
        agents=[writer, editor, fact_checker],
        # Specify that the writer should start
        selection_strategy=SequentialSelectionStrategy(initial_agent=writer),
        # Limit to 9 turns total (3 rounds × 3 agents)
        termination_strategy=DefaultTerminationStrategy(maximum_iterations=9),
    )
    await chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=input))

    # Collect responses
    responses = []
    async for response in chat.invoke():
        responses.append({"role": response.role.value, "message": response.content})

    return responses