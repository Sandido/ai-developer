import os
import asyncio
from enum import Enum
import logging
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import kernel_function
from semantic_kernel.kernel_pydantic import KernelBaseModel
from semantic_kernel.processes.kernel_process.kernel_process_step import (
    KernelProcessStep,
)
from semantic_kernel.processes.kernel_process.kernel_process_step_context import (
    KernelProcessStepContext,
)
from semantic_kernel.processes.kernel_process.kernel_process_step_state import (
    KernelProcessStepState,
)
from semantic_kernel.processes.local_runtime.local_event import KernelProcessEvent
from semantic_kernel.processes.local_runtime.local_kernel_process import start
from semantic_kernel.processes.process_builder import ProcessBuilder

from typing import Awaitable, Callable
from semantic_kernel.filters import FilterTypes, FunctionInvocationContext

# Add Logger
logger = logging.getLogger(__name__)
load_dotenv(override=True)


# Define events for our content creation process
class ContentEvents(Enum):
    StartProcess = "startProcess"
    TopicReceived = "topicReceived"
    ContentGenerated = "contentGenerated"
    ContentReviewed = "contentReviewed"
    RevisionNeeded = "revisionNeeded"
    ProcessComplete = "processComplete"


# Define state for the content creation process
class ContentState(KernelBaseModel):
    topic: str = ""
    content: str = ""
    review: str = ""
    revision_count: int = 0
    max_revisions: int = 3

    def debug_info(self):
        return f"State: topic='{self.topic}', content_length={len(self.content)}, revision_count={self.revision_count}"


# Step to get the topic from the user
class TopicInputStep(KernelProcessStep[ContentState]):
    async def activate(self, state: KernelProcessStepState[ContentState]):
        """Activates the step and sets the state."""
        state.state = state.state or ContentState()
        self.state = state.state
        print(f"DEBUG - TopicInputStep activated: {self.state.debug_info()}")

    @kernel_function
    async def get_topic(self, context: KernelProcessStepContext):
        print("What topic would you like content about?")
        topic = input("TOPIC: ")

        self.state.topic = topic
        print(f"DEBUG - Setting topic to: '{self.state.topic}'")

        await context.emit_event(
            process_event=ContentEvents.TopicReceived, data=self.state
        )
        print(f"DEBUG - Emitted TopicReceived with state.topic = '{self.state.topic}'")


# Step to generate content
class ContentGenerationStep(KernelProcessStep[ContentState]):
    async def activate(self, state: KernelProcessStepState[ContentState]):
        """Activates the step and sets the state."""
        state.state = state.state or ContentState()
        self.state = state.state
        print(f"DEBUG - ContentGenerationStep activated: {self.state.debug_info()}")

    @kernel_function
    async def generate_content(
        self,
        context: KernelProcessStepContext,
        kernel: Kernel,
        content_state: ContentState = None,
    ):
        """Generates content based on the topic."""
        if content_state:
            print(
                f"DEBUG - Received content_state parameter with topic: '{content_state.topic}'"
            )
            self.state = content_state

        print(f"Generating content about: {self.state.topic}")

        # Get chat completion service and generate content
        chat_service = kernel.get_service(service_id="process-framework")
        settings = chat_service.instantiate_prompt_execution_settings(
            service_id="process-framework"
        )

        chat_history = ChatHistory()
        chat_history.add_system_message(
            f"You are a content creator specializing in {self.state.topic}."
        )
        chat_history.add_user_message(
            f"Write a short article about {self.state.topic}. Keep it concise but informative."
        )

        response = await chat_service.get_chat_message_contents(
            chat_history=chat_history, settings=settings
        )

        if response is None:
            raise ValueError(
                "Failed to get a response from the chat completion service."
            )

        self.state.content = response[0].content

        print("Content generated successfully!")
        print(f"DEBUG - Generated content length: {len(self.state.content)}")

        await context.emit_event(
            process_event=ContentEvents.ContentGenerated, data=self.state
        )
        print(f"DEBUG - Emitted ContentGenerated with state: {self.state.debug_info()}")


# Step to review the content
class ContentReviewStep(KernelProcessStep[ContentState]):
    async def activate(self, state: KernelProcessStepState[ContentState]):
        """Activates the step and sets the state."""
        state.state = state.state or ContentState()
        self.state = state.state
        print(f"DEBUG - ContentReviewStep activated: {self.state.debug_info()}")

    @kernel_function
    async def review_content(
        self,
        context: KernelProcessStepContext,
        kernel: Kernel,
        content_state: ContentState = None,
    ):
        """Reviews the generated content."""
        if content_state:
            print(
                f"DEBUG - Received content_state parameter with topic: '{content_state.topic}'"
            )
            self.state = content_state

        print("Reviewing the generated content...")

        # Get chat completion service and review content
        chat_service = kernel.get_service(service_id="process-framework")
        settings = chat_service.instantiate_prompt_execution_settings(
            service_id="process-framework"
        )

        chat_history = ChatHistory()
        chat_history.add_system_message(
            "You are a content reviewer. Your job is to review content and provide feedback."
        )
        chat_history.add_user_message(
            f"Review this content about {self.state.topic}:\n\n{self.state.content}\n\nProvide a brief review and state whether it needs revision or is good to publish."
        )

        response = await chat_service.get_chat_message_contents(
            chat_history=chat_history, settings=settings
        )

        if response is None:
            raise ValueError(
                "Failed to get a response from the chat completion service."
            )

        self.state.review = response[0].content

        print(f"Review: {self.state.review}")

        # Check if revision is needed
        needs_revision = (
            "revision" in self.state.review.lower()
            and self.state.revision_count < self.state.max_revisions
        )

        if needs_revision:
            self.state.revision_count += 1
            print(
                f"Revision needed. Revision count: {self.state.revision_count}/{self.state.max_revisions}"
            )
            await context.emit_event(
                process_event=ContentEvents.RevisionNeeded, data=self.state
            )
            print(
                f"DEBUG - Emitted RevisionNeeded with state: {self.state.debug_info()}"
            )
        else:
            print("Content approved or max revisions reached!")
            await context.emit_event(
                process_event=ContentEvents.ContentReviewed, data=self.state
            )
            print(
                f"DEBUG - Emitted ContentReviewed with state: {self.state.debug_info()}"
            )


# Step to finalize the process
class FinalizeStep(KernelProcessStep[ContentState]):
    async def activate(self, state: KernelProcessStepState[ContentState]):
        state.state = state.state or ContentState()
        self.state = state.state
        print(f"DEBUG - FinalizeStep activated: {self.state.debug_info()}")

    @kernel_function
    async def finalize(
        self, context: KernelProcessStepContext, content_state: ContentState = None
    ):
        """Finalizes the process and displays the results."""
        if content_state:
            print(
                f"DEBUG - Received content_state parameter with topic: '{content_state.topic}'"
            )
            self.state = content_state

        print("\n" + "=" * 50)
        print("CONTENT CREATION PROCESS COMPLETE")
        print("=" * 50)
        print(f"Topic: {self.state.topic}")
        print("-" * 50)
        print("Final Content:")
        print(self.state.content)
        print("-" * 50)
        print("Review:")
        print(self.state.review)
        print("-" * 50)
        print(f"Revisions: {self.state.revision_count}")
        print("=" * 50 + "\n")

        await context.emit_event(process_event=ContentEvents.ProcessComplete, data=None)
        
def create_kernel_with_service(service_id="default"):
    """Create a kernel with Azure OpenAI or OpenAI service."""
    kernel = Kernel()

    if (os.getenv("AZURE_OPENAI_ENDPOINT")):
        print("Using Azure OpenAI service")
        kernel.add_service(
            AzureChatCompletion(service_id=service_id)
        )
    else:
        raise ValueError(
            "No AI service credentials found. Please set up your .env file."
        )

    return kernel

        
# Function to run the content creation process
async def run_content_creation_process():
    # Create our kernel
    kernel = create_kernel_with_service(service_id="process-framework")
    print("Kernel created successfully!")

    # Create a process builder
    process = ProcessBuilder(name="ContentCreation")

    # Define the steps
    topic_step = process.add_step(TopicInputStep)
    generation_step = process.add_step(ContentGenerationStep)
    review_step = process.add_step(ContentReviewStep)
    finalize_step = process.add_step(FinalizeStep)

    # Define the input event that starts the process
    process.on_input_event(event_id=ContentEvents.StartProcess).send_event_to(
        target=topic_step
    )

    # Define the event flow with explicit parameter mapping
    topic_step.on_event(event_id=ContentEvents.TopicReceived).send_event_to(
        target=generation_step, parameter_name="content_state"
    )
    generation_step.on_event(event_id=ContentEvents.ContentGenerated).send_event_to(
        target=review_step, parameter_name="content_state"
    )
    review_step.on_event(event_id=ContentEvents.RevisionNeeded).send_event_to(
        target=generation_step, parameter_name="content_state"
    )
    review_step.on_event(event_id=ContentEvents.ContentReviewed).send_event_to(
        target=finalize_step, parameter_name="content_state"
    )
    finalize_step.on_event(event_id=ContentEvents.ProcessComplete).stop_process()

    # Build the kernel process
    kernel_process = process.build()

    # Start the process
    await start(
        process=kernel_process,
        kernel=kernel,
        initial_event=KernelProcessEvent(id=ContentEvents.StartProcess, data=None),
    )


if __name__ == "__main__":
    # Run the content creation process
    asyncio.run(run_content_creation_process())