import os
from typing import TypedDict, Annotated, Literal, List
from enum import Enum
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.memory.azure_ai_search import AzureAISearchCollection, AzureAISearchStore
from semantic_kernel.data.vector_search import VectorSearchOptions
from semantic_kernel import Kernel

from models.employee_handbook_model import EmployeeHandbookModel
from openai import AzureOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator, ConfigDict
import logging

from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.agents import ChatHistoryAgentThread


from typing import Annotated
from pydantic import BaseModel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.functions import kernel_function, KernelArguments

# Add Logger
logger = logging.getLogger(__name__)


load_dotenv(override=True)

class OpenAIConfig(BaseModel):
    """Configuration for Azure OpenAI."""
    deployment_name: str = Field(..., description="The deployment name for Azure OpenAI service")
    api_key: str = Field(..., description="The API key for Azure OpenAI service")
    azure_endpoint: str = Field(..., description="The base URL for Azure OpenAI resource")
    api_version: str = Field(..., description="The API version to use")

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        return cls(
            deployment_name=os.environ['AZURE_OPENAI_CHAT_DEPLOYMENT_NAME'],
            api_key=os.environ['AZURE_OPENAI_API_KEY'],
            azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
            api_version=os.environ['AZURE_OPENAI_API_VERSION']
        )
        
class RenameModel(BaseModel):
    animal: str
    latin_animal: str
    
settings = OpenAIChatPromptExecutionSettings()
settings.response_format = RenameModel

class RenamerPlugin:
    """Renamer Plugin to rename files and folders."""
    def __init__(self, kernel: Kernel):
        logger.info("Translation Plugin init !!!!!!!!!!!! logger")
        self.config = OpenAIConfig.from_env()
        self.openai_client = AzureOpenAI(
            api_key=self.config.api_key,
            azure_endpoint=self.config.azure_endpoint,
            api_version=self.config.api_version
        )

    @kernel_function(description="Change the animal name the user provided to the latin version of the name.", name="latin_rename_animal")
    async def latin_rename_animal(self, query_str: Annotated[str, "Rename the user provided animal name to its latin name. Add no other text."]) -> Annotated[str, "Latin name of the animal."]:
        logger.info("Animal Renamer!!!!!!!!!!!! logger")
        simple_agent = ChatCompletionAgent(
            service=AzureChatCompletion(),
            name="renamer_assistant",
            instructions="Only translate the animal word to its latin version. Add no other text.",
            arguments=KernelArguments(settings),
        )
        
        
        ## example below of how to use chathistoryThreadAgent. 
        # Not needed as my ChatHistory works much better.
        thread: ChatHistoryAgentThread = None
        
        user_messages = [
            "lion",
            "manatee",
            "elephant",
        ]

        for user_message in user_messages:
            print("*** User:", user_message)
            
            # get our response from the agent
            response = await simple_agent.get_response(messages=user_message, thread=thread)
            print("*** Agent:", response.content)
            
            # save the thread with all the existing messages and responses
            thread = response.thread

        # print the final conversation, so we can see what happens in thread
        print("-" * 25)
        async for m in thread.get_messages():
            print(m.role, m.content)
        ## Example above
            
            
        
        response = await simple_agent.get_response(messages=query_str)
        print("Agent RESPONSE:", response.content)
        return response.content