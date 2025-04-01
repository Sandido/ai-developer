import asyncio
import logging
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAITextToImage, AzureTextEmbedding
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.openapi_plugin import OpenAPIFunctionExecutionParameters
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions import KernelArguments

from plugins.ai_search_plugin import AiSearchPlugin
from openai import AzureOpenAI
import os

# Add Logger
logger = logging.getLogger(__name__)

load_dotenv(override=True)

chat_history = ChatHistory()

def initialize_kernel():
    kernel = Kernel()
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

    kernel.add_plugin(AiSearchPlugin(kernel), plugin_name="AISearch") 

    return kernel


async def process_message(user_input):
    logger.info(f"Processing user message: {user_input}")
    kernel = initialize_kernel()
    chat_function = kernel.add_function(
        prompt="{{$chat_history}}{{$user_input}}",
        plugin_name="ChatBot",
        function_name="Chat",
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