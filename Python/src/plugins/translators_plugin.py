"""
Translator Plugin for Semantic Kernel.
Using multiple different agents for different languages. 
Can auto detect which agent to use well, and responds to specific language translation requests.
"""

import os
from typing import Annotated, List
from enum import Enum
from semantic_kernel.functions import kernel_function
from semantic_kernel import Kernel

from openai import AzureOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
import logging
# Add Logger
logger = logging.getLogger(__name__)


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
        logger.info("Translation Plugin init !!!!!!!!!!!! logger")
        self.config = OpenAIConfig.from_env()
        self.openai_client = AzureOpenAI(
            api_key=self.config.api_key,
            azure_endpoint=self.config.azure_endpoint,
            api_version=self.config.api_version
        )

    @kernel_function(description="Translates spanish text to english. Use this when spanish text is provided.", name="translate_spanish_to_english")
    async def translate_spanish_to_english(self, query_str: Annotated[str, "Strings in spanish to be translated to english. Add nothing new."]) -> Annotated[str, "Translated string in english"]:
        logger.info("Translate spanish to english!!!!!!!!!!!! logger")
        translation_request = TranslationRequest(
            text=query_str,
            source_language="Spanish",
            target_language="English"
        )

        return await self.translate(translation_request)

    @kernel_function(description="Translates japanese text to english. Use this when japanese text is provided.", name="translate_japanese_to_english")
    async def translate_japanese_to_english(self, query_str: Annotated[str, "Strings in japanese to be translated to english. Add nothing new."]) -> Annotated[str, "Translated string in english"]:
        logger.info("Translate japanese to english!!!!!!!!!!!! logger")
        translation_request = TranslationRequest(
            text=query_str,
            source_language="Japanese",
            target_language="English"
        )

        return await self.translate(translation_request)

    @kernel_function(description="Translates hindi text to english. Use this when hindi text is provided.", name="translate_hindi_to_english")
    async def translate_hindi_to_english(self, query_str: Annotated[str, "Strings in hindi to be translated to english. Add nothing new."]) -> Annotated[str, "Translated string in english"]:
        logger.info("Translate hindi to english!!!!!!!!!!!! logger")
        translation_request = TranslationRequest(
            text=query_str,
            source_language="Hindi",
            target_language="English"
        )

        return await self.translate(translation_request)

    @kernel_function(description="Only translates text between specified languages when 2 languages are provided by the user. Add nothing new.", name="translate")
    async def translate(self, request: TranslationRequest) -> str:
        logger.info("Generic Translate call!!!!!!!!!!!! logger")
        """Translate text between languages."""
        # Create structured messages using our new models
        messages = [
            ChatMessage.system(f"You are a translator from {request.source_language} to {request.target_language}. Add nothing new. Translate the following text."),
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