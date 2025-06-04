import os
import asyncio
from enum import Enum

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

# Define the events our process will use
class HelloWorldEvents(Enum):
    StartProcess = "startProcess"
    NameReceived = "nameReceived"
    ProcessComplete = "processComplete"


# Define the state for our process
class HelloWorldState(KernelBaseModel):
    name: str = ""
    greeting: str = ""


# Step 1: Get the user's name
class GetNameStep(KernelProcessStep[HelloWorldState]):
    def create_default_state(self) -> HelloWorldState:
        """Creates the default HelloWorldState."""
        return HelloWorldState()

    async def activate(self, state: KernelProcessStepState[HelloWorldState]):
        """Initialize the step's state when activated."""
        self.state = state.state or self.create_default_state()
        print("GetNameStep activated")

    @kernel_function(name="get_name")
    async def get_name(self, context: KernelProcessStepContext):
        """Get the user's name."""
        print("What is your name?")
        name = input("Name: ")

        # Store the name in our state
        self.state.name = name
        print(f"Name set to: {self.state.name}")

        # Emit an event to signal that we have the name
        await context.emit_event(
            process_event=HelloWorldEvents.NameReceived, data=self.state
        )


# Step 2: Display the greeting
class DisplayGreetingStep(KernelProcessStep[HelloWorldState]):
    async def activate(self, state: KernelProcessStepState[HelloWorldState]):
        """Initialize the step's state when activated."""
        self.state = state.state or HelloWorldState()
        print("DisplayGreetingStep activated")

    @kernel_function(name="display_greeting")
    async def display_greeting(
        self, context: KernelProcessStepContext, hello_state: HelloWorldState = None
    ):
        """Display the greeting and complete the process."""
        # If we received state from the event, use it
        if hello_state:
            self.state = hello_state

        # Generate the greeting
        self.state.greeting = f"Hello, {self.state.name}! Welcome to the Semantic Kernel Process Framework."

        # Display the greeting with some decoration
        print("\n" + "=" * 50)
        print(self.state.greeting)
        print("=" * 50 + "\n")

        # Emit an event to signal that the process is complete
        await context.emit_event(
            process_event=HelloWorldEvents.ProcessComplete, data=None
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


# Function to run the process
async def run_hello_world_process():
    # Create a kernel
    kernel = Kernel()
    kernel.add_filter(FilterTypes.FUNCTION_INVOCATION, function_invocation_filter)
    # Create a process builder
    process = ProcessBuilder(name="HelloWorld")

    # Add the steps to the process
    name_step = process.add_step(GetNameStep)
    greeting_step = process.add_step(DisplayGreetingStep)

    # Define the process flow using events
    # 1. The StartProcess event triggers the GetNameStep
    process.on_input_event(event_id=HelloWorldEvents.StartProcess).send_event_to(
        target=name_step
    )

    # 2. When name is received, send to greeting generation step
    name_step.on_event(event_id=HelloWorldEvents.NameReceived).send_event_to(
        target=greeting_step, parameter_name="hello_state"
    )

    # 3. When process is complete, stop the process
    greeting_step.on_event(event_id=HelloWorldEvents.ProcessComplete).stop_process()

    # Build the process
    kernel_process = process.build()

    # Start the process with the initial event
    await start(
        process=kernel_process,
        kernel=kernel,
        initial_event=KernelProcessEvent(id=HelloWorldEvents.StartProcess, data=None),
    )
    
    
if __name__ == "__main__":
    # Run the process
    asyncio.run(run_hello_world_process())