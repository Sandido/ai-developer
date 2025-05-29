import asyncio
import logging
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions import KernelArguments

from plugins.ai_search_plugin import AiSearchPlugin
from plugins.translators_plugin import TranslatorPlugins
from plugins.weather_plugins import WeatherPlugin
from plugins.renamer_plugin import RenamerPlugin
from plugins.tournament_plugin import TournamentPlugin
from openai import AzureOpenAI
import os

from typing import Awaitable, Callable
from semantic_kernel.filters import FilterTypes, FunctionInvocationContext

# Add Logger
logger = logging.getLogger(__name__)

load_dotenv(override=True)

chat_history = ChatHistory()

from typing import Callable, Optional
def initialize_kernel(notify: Optional[Callable[[str], None]] = None):
    kernel = Kernel()
    # kernel.add_filter(FilterTypes.FUNCTION_INVOCATION, function_invocation_filter)

    chat_completion_service = AzureChatCompletion(service_id="chat-completion")
    kernel.add_service(chat_completion_service)
    logger.info("Chat completion service added to the kernel.")

    embedding_service = AzureTextEmbedding(
                api_key = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
                endpoint =os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT") ,
                deployment_name=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT_NAME"),
                service_id="embedding"
                )
    kernel.add_service(embedding_service)

    # kernel.add_plugin(AiSearchPlugin(kernel), plugin_name="AISearch")

    kernel.add_plugin(TranslatorPlugins(kernel), plugin_name="TranslatorPlugins", description="Translator Plugin to translate only non-english text.")
    try:
        kernel.add_plugin(RenamerPlugin(kernel), plugin_name="RenamerPlugin", description="Animal Renamer Plugin to change an animal name to its latin name.")
    except Exception as e:
        logger.error(f"Error adding RenamerPlugin: {e}")
        print("Error adding RenamerPlugin:", e)
        
    print(" Renamer Plugin added to the kernel.")
    kernel.add_plugin(WeatherPlugin(kernel), plugin_name="WeatherPlugin", description="Weather Plugin to answer anything about Weather user inputs.")

    print("adding tournament plugin")
    kernel.add_plugin(TournamentPlugin(kernel, notify=notify), plugin_name="TournamentPlugin", description="Tournament Plugin to run a tournament for a fighter. Start when a fighter is present.")

    return kernel

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

async def process_message(user_input, notify=None):
    logger.info(f"Processing user message: {user_input}")
    kernel = initialize_kernel(notify)

    chat_function = kernel.add_function(
        prompt=( # seems like adding a system prompt helps prevent plugins from being used correctly.
            "{{ $chat_history }}\n"
            "User: {{ $user_input }}\n"
            "Assistant:"
        ),
        plugin_name="ChatBot",
        function_name="Chat",
        description="General chat; model may call other tools.",
    )

    execution_settings = kernel.get_prompt_execution_settings_from_service_id("chat-completion")
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    arguments = KernelArguments(settings=execution_settings)
    arguments["user_input"] = user_input
    arguments["chat_history"] = chat_history
    result = await kernel.invoke(chat_function, arguments=arguments)

    chat_history.add_user_message(user_input)
    chat_history.add_assistant_message(str(result))
    return result

def reset_chat_history():
    global chat_history
    chat_history = ChatHistory()