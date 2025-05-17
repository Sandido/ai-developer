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

load_dotenv(override=True)

class ChatRole(str, Enum):
    """Valid roles for chat messages."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"

class ChatMessage(BaseModel):
    """Model for a single chat message."""
    role: ChatRole
    content: str
    
    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        """Create a system message."""
        return cls(role=ChatRole.SYSTEM, content=content)
    
    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        """Create a user message."""
        return cls(role=ChatRole.USER, content=content)
    
    @classmethod
    def assistant(cls, content: str) -> "ChatMessage":
        """Create an assistant message."""
        return cls(role=ChatRole.ASSISTANT, content=content)

class ChatCompletionRequest(BaseModel):
    """Model for chat completion requests."""
    model: str
    messages: List[ChatMessage]
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=None)

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

class TranslationRequest(BaseModel):
    print("TranslationRequest model!!!!!!!!!!!!")
    """Model for translation requests."""
    text: str = Field(..., description="Text to be translated")
    source_language: str = Field(..., description="Source language")
    target_language: str = Field(..., description="Target language")
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "Hola, ¿cómo estás?",
                    "source_language": "Spanish",
                    "target_language": "English"
                }
            ]
        }
    )

class TranslationResponse(BaseModel):
    """Model for translation responses."""
    translated_text: str = Field(..., description="Translated text")
    source_language: str = Field(..., description="Source language")
    target_language: str = Field(..., description="Target language")

class TranslatorPlugins:
    def __init__(self, kernel: Kernel):
        print("Translation Plugin init !!!!!!!!!!!!")
        self.config = OpenAIConfig.from_env()
        self.openai_client = AzureOpenAI(
            api_key=self.config.api_key,  
            azure_endpoint=self.config.azure_endpoint,
            api_version=self.config.api_version
        )
        
    @kernel_function(description="Translates any text in spanish to english.", name="translate_spanish_to_english")
    async def translate_spanish_to_english(self, query_str: Annotated[str, "Strings in spanish to be translated to english."]) -> Annotated[str, "Translated string in english"]:
        print("TranslationRequest spanish to english!!!!!!!!!!!!")
        translation_request = TranslationRequest(
            text=query_str,
            source_language="Spanish",
            target_language="English"
        )
        
        return await self.translate(translation_request)
    
    @kernel_function(description="Translates any text in japanese to english.", name="translate_japanese_to_english")
    async def translate_japanese_to_english(self, query_str: Annotated[str, "Strings in japanese to be translated to english."]) -> Annotated[str, "Translated string in english"]:
        print("TranslationRequest japanese to english!!!!!!!!!!!!")
        translation_request = TranslationRequest(
            text=query_str,
            source_language="Japanese",
            target_language="English"
        )
        
        return await self.translate(translation_request)
    
    @kernel_function(description="Translates any text in hindi to english.", name="translate_hindi_to_english")
    async def translate_hindi_to_english(self, query_str: Annotated[str, "Strings in hindi to be translated to english."]) -> Annotated[str, "Translated string in english"]:
        print("Translate hindi to english!!!!!!!!!!!!")
        translation_request = TranslationRequest(
            text=query_str,
            source_language="Hindi",
            target_language="English"
        )
        
        return await self.translate(translation_request)
    
    @kernel_function(description="Translates text between specified languages.", name="translate")
    async def translate(self, request: TranslationRequest) -> str:
        print("Generic Translate call!!!!!!!!!!!!")
        """Translate text between languages."""
        # Create structured messages using our new models
        messages = [
            ChatMessage.system(f"You are a translator from {request.source_language} to {request.target_language}. Translate the following text."),
            ChatMessage.user(request.text)
        ]
        # Create a chat completion request
        chat_request = ChatCompletionRequest(
            model=self.config.deployment_name,
            messages=messages
        )
        # Convert to dict for the API call
        response = self.openai_client.chat.completions.create(**chat_request.model_dump())
        translation = response.choices[0].message.content
        
        result = TranslationResponse(
            translated_text=translation,
            source_language=request.source_language,
            target_language=request.target_language
        )
        return result.translated_text