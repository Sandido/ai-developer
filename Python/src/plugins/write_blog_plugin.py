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
from semantic_kernel.agents import ChatCompletionAgent, AgentGroupChat
from semantic_kernel.functions import kernel_function, KernelArguments
from semantic_kernel.agents.strategies import DefaultTerminationStrategy, SequentialSelectionStrategy


# Add Logger
logger = logging.getLogger(__name__)


load_dotenv(override=True)

class WriteBlogPlugin:

    def __init__(self, kernel: Kernel):
        if not kernel.get_service("embedding"):
            raise Exception("Missing AI Foundry embedding service")
        self.client = kernel.get_service("embedding")

    researcher = ChatCompletionAgent(
        service=AzureChatCompletion(),
        name="Researcher",
        instructions="""You are a thorough researcher who provides factual information and analysis.
        
        Your responsibilities:
        - Provide accurate, factual information on the topic
        - Analyze questions from multiple angles
        - Consider historical context and current understanding
        - Be objective and balanced in your assessment
        - Acknowledge limitations in current knowledge when appropriate
        
        Always strive for accuracy over speculation. Keep your responses very concise, clear and straightforward.""",
    )

    # 2. Creative thinker who generates innovative ideas
    innovator = ChatCompletionAgent(
        service=AzureChatCompletion(),
        name="Innovator",
        instructions="""You are an innovative thinker who generates novel ideas and perspectives.
        
        Your responsibilities:
        - Suggest unique approaches and solutions
        - Think beyond conventional boundaries
        - Make unexpected connections between concepts
        - Offer imaginative scenarios and possibilities
        - Propose 'what if' scenarios to expand thinking
        
        Don't be constrained by traditional thinking - be bold and creative. Keep your responses very concise, imaginative and engaging.""",
    )

    # 3. Critic who evaluates ideas critically
    critic = ChatCompletionAgent(
        service=AzureChatCompletion(),
        name="Critic",
        instructions="""You are a thoughtful critic who evaluates ideas and identifies potential issues.
        
        Your responsibilities:
        - Analyze the strengths and weaknesses of proposals
        - Identify potential problems or limitations
        - Challenge assumptions constructively
        - Suggest improvements to ideas
        - Consider practical implementation challenges
        
        Be constructive in your criticism - your goal is to improve ideas, not dismiss them. Keep your responses very concise, clear and straightforward.""",
    )

    # 4. Synthesizer who brings ideas together
    synthesizer = ChatCompletionAgent(
        service=AzureChatCompletion(),
        name="Synthesizer",
        instructions="""You are a skilled synthesizer who integrates diverse perspectives into coherent conclusions.
        
        Your responsibilities:
        - Identify common themes across different viewpoints
        - Reconcile apparently conflicting ideas when possible
        - Create a balanced, integrated perspective
        - Summarize key points from the discussion
        - Draw reasonable conclusions from the collective input
        
        Your goal is to bring together different perspectives into a coherent whole. Keep your responses very concise, clear and straightforward.""",
    )

    # Create a group chat with all specialized agents
    expert_team = AgentGroupChat(
        agents=[researcher, innovator, critic, synthesizer],
        selection_strategy=SequentialSelectionStrategy(),
        termination_strategy=DefaultTerminationStrategy(
            maximum_iterations=8
        ),  # 2 rounds of all 4 agents
    )