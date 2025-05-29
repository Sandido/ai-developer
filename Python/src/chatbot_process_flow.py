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


from typing import Awaitable, Callable
from semantic_kernel.filters import FilterTypes, FunctionInvocationContext

# Add Logger
logger = logging.getLogger(__name__)
load_dotenv(override=True)

# Create a step to handle the introduction
class IntroStep(KernelProcessStep):
    @kernel_function
    async def print_intro_message(self):
        print("Welcome to the Semantic Kernel Process Framework Chatbot!\n")
        print("Type 'exit' to end the conversation.\n")


# Define events for our chatbot process
class ChatBotEvents(Enum):
    StartProcess = "startProcess"
    IntroComplete = "introComplete"
    UserInputReceived = "userInputReceived"
    AssistantResponseGenerated = "assistantResponseGenerated"
    Exit = "exit"


# Define state for user input step
class UserInputState(KernelBaseModel):
    user_inputs: list[str] = []
    current_input_index: int = 0


# Create a step to handle user input
class UserInputStep(KernelProcessStep[UserInputState]):
    def create_default_state(self) -> "UserInputState":
        """Creates the default UserInputState."""
        return UserInputState()

    async def activate(self, state: KernelProcessStepState[UserInputState]):
        """Activates the step and sets the state."""
        state.state = state.state or self.create_default_state()
        self.state = state.state

    @kernel_function(name="get_user_input")
    async def get_user_input(self, context: KernelProcessStepContext):
        """Gets the user input."""
        if not self.state:
            raise ValueError("State has not been initialized")

        user_message = input("USER: ")

        print(user_message)

        if "exit" in user_message:
            await context.emit_event(process_event=ChatBotEvents.Exit, data=None)
            return

        self.state.current_input_index += 1

        # Emit the user input event
        await context.emit_event(
            process_event=ChatBotEvents.UserInputReceived, data=user_message
        )


# Define state for the chatbot response step
class ChatBotState(KernelBaseModel):
    chat_messages: list = []


# Create a step to handle the chatbot response
class ChatBotResponseStep(KernelProcessStep[ChatBotState]):
    state: ChatBotState = None

    async def activate(self, state: KernelProcessStepState[ChatBotState]):
        """Activates the step and initializes the state object."""
        self.state = state.state or ChatBotState()
        self.state.chat_messages = self.state.chat_messages or []

    @kernel_function(name="get_chat_response")
    async def get_chat_response(
        self, context: KernelProcessStepContext, user_message: str, kernel: Kernel
    ):
        """Generates a response from the chat completion service."""
        # Add user message to the state
        self.state.chat_messages.append({"role": "user", "message": user_message})

        # Get chat completion service and generate a response
        chat_service = kernel.get_service(service_id="process-framework")
        settings = chat_service.instantiate_prompt_execution_settings(
            service_id="process-framework"
        )

        chat_history = ChatHistory() # TODO this chat history doesn't work, remade each message, tsk example tsk.
        chat_history.add_user_message(user_message)
        response = await chat_service.get_chat_message_contents(
            chat_history=chat_history, settings=settings
        )

        if response is None:
            raise ValueError(
                "Failed to get a response from the chat completion service."
            )

        answer = response[0].content

        print(f"ASSISTANT: {answer}")

        # Update state with the response
        self.state.chat_messages.append(answer)

        # Emit an event: assistantResponse
        await context.emit_event(
            process_event=ChatBotEvents.AssistantResponseGenerated, data=answer
        )

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


# Create our kernel
kernel = create_kernel_with_service(service_id="process-framework")
# kernel.add_filter(FilterTypes.FUNCTION_INVOCATION, function_invocation_filter)
print("Kernel created successfully!")



# Function to run the chatbot process
async def run_chatbot_process():
    # Create a process builder
    process = ProcessBuilder(name="ChatBot")

    # Define the steps
    intro_step = process.add_step(IntroStep)
    user_input_step = process.add_step(UserInputStep)
    response_step = process.add_step(ChatBotResponseStep)

    # Define the input event that starts the process and where to send it
    process.on_input_event(event_id=ChatBotEvents.StartProcess).send_event_to(
        target=intro_step
    )

    # Define the event that triggers the next step in the process
    intro_step.on_function_result(
        function_name=IntroStep.print_intro_message.__name__
    ).send_event_to(target=user_input_step)

    # Define the event that triggers the process to stop
    user_input_step.on_event(event_id=ChatBotEvents.Exit).stop_process()

    # For the user step, send the user input to the response step
    user_input_step.on_event(event_id=ChatBotEvents.UserInputReceived).send_event_to(
        target=response_step, parameter_name="user_message"
    )

    # For the response step, send the response back to the user input step
    response_step.on_event(
        event_id=ChatBotEvents.AssistantResponseGenerated
    ).send_event_to(target=user_input_step)

    # Build the kernel process
    kernel_process = process.build()

    # Start the process
    await start(
        process=kernel_process,
        kernel=kernel,
        initial_event=KernelProcessEvent(id=ChatBotEvents.StartProcess, data=None),
    )


# Run the chatbot process
if __name__ == "__main__":
    asyncio.run(run_chatbot_process())

# so what's happening here is that the Semantic Kernel idea of dynamically selecting methods
# is being applied to the methods in a class now.
# The Classes are manually setup in a node flow with the events, but each event is given
# a certain parameter. Then the Semantic Kernel decision comes down to which method
# in the manually selected class is run depending on what best fits the parameters
# given by the previous user input or event.
# The benefits here are great. Now instead of lots of Agents providing input
# in less specific orders and sometimes chaotically, we can have a more
# structured process, like an actual workflow, but each agent dyanamically chooses
# its actions to do.