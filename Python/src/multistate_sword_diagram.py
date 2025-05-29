
import os
import asyncio
from enum import Enum
import logging
from dotenv import load_dotenv
import mermaid as md
from mermaid.graph import Graph

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

from mermaid_cli import render_mermaid

DIAGRAM = """
flowchart LR
    A((Start)) --> R[Registration]
    R --> L[Lines]
    L --> B["Bullpen Desk"]

    %% decision paths
    B --Approved--> W["Pre-Fight Waiting Area"]
    B --Rejected--> P["Resolve Issues"]

    %% loop back for re-check
    P --> B

    %% toward the ring
    W --> F["Fight Rings"]
    F --Injury (1%)--> M[Medic]

    %% ends
    F --> Z((End))
    M --> Z
"""



async def main() -> None:
    # Generate SVG (change to "png" or "pdf" if you like)
    _, _, svg_bytes = await render_mermaid(DIAGRAM, output_format="svg")
    out_file = "tournament_flow.svg"
    with open(out_file, "wb") as f:
        f.write(svg_bytes)
    print(f"Diagram saved to {out_file}")

if __name__ == "__main__":
    asyncio.run(main())