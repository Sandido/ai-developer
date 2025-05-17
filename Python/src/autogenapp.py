from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import os
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

load_dotenv(override=True)

DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Create an OpenAI model client
# client = OpenAIChatCompletionClient(model=DEPLOYMENT_NAME)
client = AzureOpenAIChatCompletionClient(
    model=DEPLOYMENT_NAME,
    api_version=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY, # For key-based authentication.
)


# Create the writer agent
writer = AssistantAgent("writer", model_client=client, system_message="Draft a short paragraph on climate change.")

# Create the reviewer agent
reviewer = AssistantAgent("reviewer", model_client=client, system_message="Review the draft and suggest improvements.")

# Build the graph
builder = DiGraphBuilder()
builder.add_node(writer).add_node(reviewer)
builder.add_edge(writer, reviewer)

# Build and validate the graph
graph = builder.build()

# Create the flow
flow = GraphFlow([writer, reviewer], graph=graph)
