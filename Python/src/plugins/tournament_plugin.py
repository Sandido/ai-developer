"""
This agent models a fencing tournament flow, where a fighter goes through
various steps from registration to the final fight, with potential gear checks
and injury handling. The flow is designed to be interactive, allowing the
fighter to input their name, gear, and respond to prompts at each step.

This is to learn the semantic_kernel.processes module and how to build
an interactive process that is EXPLICITLY and RIGIDLY mapped out with different Agents. 
"""

from typing import TypedDict, Annotated, Optional
import requests
from semantic_kernel.functions import kernel_function
import os
from dotenv import load_dotenv

import os, asyncio, logging, random
from enum import Enum
from typing import Callable, Optional

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
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

from semantic_kernel.contents import ChatHistory
from semantic_kernel import Kernel
from typing import ClassVar, List
from semantic_kernel.functions import kernel_function

logger = logging.getLogger(__name__)
load_dotenv(override=True)

class TournamentPlugin:
    def __init__(self, kernel: Kernel, notify: Callable[[str], None] | None = None):
        self.kernel = kernel
        self.notify = notify or (lambda txt: print(txt))
        logger.info("Tournament Plugin initialized")

    @kernel_function(description="Starts the tournament flow process.")
    async def start_tournament_flow(self): # notify=lambda txt: print(txt)):
        print("Starting the tournament flow process...")
        summary = await run_tournament_flow(notify=self.notify)
        print("Tournament flow process completed. {}".format(summary))
        return summary or "Tournament finished."


# ─────────────────────────────────────────────────────────────────────────────
# EVENTS ─ mirrors the Mermaid edges
class FighterEvents(str, Enum):
    StartProcess     = "startProcess"
    RegistrationDone = "registrationDone"
    LinesCleared     = "linesCleared"
    Approved         = "approved"
    Rejected         = "rejected"
    GearIssuesResolved   = "gearIssuesResolved"
    WaitingOver      = "waitingOver"
    FightOver        = "fightOver"
    Injury           = "injury"
    ProcessComplete  = "processComplete"

# ─────────────────────────────────────────────────────────────────────────────
# STATE  (can be expanded later)
class FighterState(KernelBaseModel):
    fighter_name: str = ""
    gear_notes: str = ""
    approved: bool = False
    injury: bool = False
    injury_occurred: bool = False
    summary: str = ""
    journey: list[str] = []
    notify: Optional[Callable[[str], None]] = None

    def say(self, text: str):
        """Send text to UI and console."""
        if self.notify:
            self.notify(text)
        print(text)

    def debug(self) -> str:
        return f"(name={self.fighter_name!r}, approved={self.approved}, injury={self.injury})"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1  – Registration
class RegistrationStep(KernelProcessStep[FighterState]):
    async def activate(self, step_state: KernelProcessStepState[FighterState]):
        step_state.state = step_state.state or FighterState()
        self.state = step_state.state
        print(f"[DEBUG] Registration activated {self.state.debug()}")
        self.state.say("Registration started.")

    @kernel_function
    async def register(self, ctx: KernelProcessStepContext, state: FighterState):
        self.state = state
        self.state.fighter_name = input("Fighter name: ").strip()
        self.state.journey.append("RegistrationStep → RegistrationDone")
        print("✔  Registration complete.")
        self.state.say("✔  Registration complete.")
        await ctx.emit_event(process_event=FighterEvents.RegistrationDone, data=self.state)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Lines
class LinesStep(KernelProcessStep[FighterState]):
    async def activate(self, st: KernelProcessStepState[FighterState]):
        self.state = st.state or FighterState()
        print(f"[DEBUG] Lines activated {self.state.debug()}")

    @kernel_function
    async def wait_in_line(self, ctx: KernelProcessStepContext, state: FighterState):
        self.state = state
        self.state.journey.append("LinesStep → LinesCleared")
        input("Press ENTER once fighter reaches the Bullpen Desk… ")
        await ctx.emit_event(process_event=FighterEvents.LinesCleared, data=self.state)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Bullpen (gear check)
class BullpenStep(KernelProcessStep[FighterState]):
    REQUIRED_GEAR: ClassVar[List[str]] = [
        "helmet",
        "back of head protection",
        "sword",
        "jacket",
        "gorget",
        "elbow protection",
        "gloves",
        "shin protection",
        "cup",
        "shoes",
        "pants",
    ]

    async def activate(self, st: KernelProcessStepState[FighterState]):
        self.state = st.state or FighterState()
        print(f"[DEBUG] Bullpen activated {self.state.debug()}")

    @kernel_function
    async def gear_check(                     # kernel is auto-injected by SK
        self,
        ctx: KernelProcessStepContext,
        kernel: Kernel,
        state: FighterState
    ):
        self.state = state
        self.state.journey.append("BullpenStep → Getting Gear Checked")
        gear_input = input(
            "Enter fighter gear (comma-separated): "
        ).strip()

        # --- build LLM prompt ---------------------------------------------
        chat = kernel.get_service(service_id="process-framework")
        settings = chat.instantiate_prompt_execution_settings(
            service_id="process-framework"
        )

        chat_history = ChatHistory()
        chat_history.add_system_message(
            "You are the head gear inspector at a HEMA tournament. "
            "You must decide if the fighter may advance. "
            "Required items (all must be present, case-insensitive): "
            f"{', '.join(self.REQUIRED_GEAR)}. "
            "Reply with:\n"
            "  APPROVE           — if ALL items are present\n"
            "  REJECT: item1, …  — if anything is missing "
            "(list missing items after the colon). "
            "Use no other words."
        )
        chat_history.add_user_message(
            f"The fighter has: {gear_input}"
        )

        resp = await chat.get_chat_message_contents(
            chat_history=chat_history, settings=settings
        )
        verdict = resp[0].content.strip().lower()
        print(f"[LLM verdict] {verdict}")

        # --- interpret verdict --------------------------------------------
        if verdict.startswith("approve"):
            self.state.approved = True
            self.state.journey.append("Gear Check → Gear Approved")
            await ctx.emit_event(process_event=FighterEvents.Approved, data=self.state)
        elif verdict.startswith("reject"):
            self.state.approved = False
            self.state.journey.append("Gear Check → Gear Rejected")
            # everything after "reject:" (if given) becomes the issue notes
            self.state.gear_notes = verdict.split(":", 1)[-1].strip()
            await ctx.emit_event(process_event=FighterEvents.Rejected, data=self.state)
        else:
            print("⚠  Unexpected model output — treating as rejection.")
            self.state.approved = False
            self.state.gear_notes = "Unclear verdict from model"
            await ctx.emit_event(process_event=FighterEvents.Rejected, data=self.state)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Resolve Issues
class ResolveIssuesStep(KernelProcessStep[FighterState]):
    async def activate(self, st: KernelProcessStepState[FighterState]):
        self.state = st.state or FighterState()
        print(f"[DEBUG] ResolveIssues activated {self.state.debug()}")

    @kernel_function
    async def fix_gear(self, ctx: KernelProcessStepContext, state: FighterState):
        self.state = state
        print(f"⚒  Resolving: {self.state.gear_notes}")
        self.state.journey.append("Resolve Issues Step → Fixed Gear Issues")
        input("Press ENTER when issues are fixed and fighter returns to Bullpen… ")
        await ctx.emit_event(process_event=FighterEvents.GearIssuesResolved, data=self.state)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Pre-Fight Waiting Area
class WaitingStep(KernelProcessStep[FighterState]):
    async def activate(self, st: KernelProcessStepState[FighterState]):
        self.state = st.state or FighterState()
        print(f"[DEBUG] WaitingArea activated {self.state.debug()}")

    @kernel_function
    async def wait(self, ctx: KernelProcessStepContext, state: FighterState):
        self.state = state
        self.state.journey.append("Waiting Step → Waiting for Fight")
        input("Waiting… (ENTER when fighter is called to ring) ")
        await ctx.emit_event(process_event=FighterEvents.WaitingOver, data=self.state)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 – Fight Rings
class FightStep(KernelProcessStep[FighterState]):
    async def activate(self, st: KernelProcessStepState[FighterState]):
        self.state = st.state or FighterState()
        print(f"[DEBUG] Fight activated {self.state.debug()}")

    @kernel_function
    async def fight(self, ctx: KernelProcessStepContext, state: FighterState):
        self.state = state
        # 1 % injury chance
        self.state.injury = random.random() < 0.01 #0.01
        if self.state.injury:
            print("⚠  Fighter injured! Sending to medic.")
            self.state.journey.append("Fighter Injured → Need a Medic")
            self.state.injury_occurred = True
            await ctx.emit_event(process_event=FighterEvents.Injury, data=self.state)
        else:
            print("✔  Fight completed without injury.")
            self.state.journey.append("Fight Completed → Ending Fight")
            await ctx.emit_event(process_event=FighterEvents.FightOver, data=self.state)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 – Medic
class MedicStep(KernelProcessStep[FighterState]):
    async def activate(self, st: KernelProcessStepState[FighterState]):
        self.state = st.state or FighterState()
        print(f"[DEBUG] Medic activated {self.state.debug()}")

    @kernel_function(description="To treat the fighter after injury to make them recover.")
    async def treat(self, ctx: KernelProcessStepContext, state: FighterState):
        self.state = state
        self.state.journey.append("Medic Step → Treating Injured Fighter")
        input("Medic treatment done (press ENTER)… ")
        await ctx.emit_event(process_event=FighterEvents.FightOver, data=self.state)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 – Finalize / End
class FinalizeStep(KernelProcessStep[FighterState]):
    async def activate(self, st: KernelProcessStepState[FighterState]):
        self.state = st.state or FighterState()
        print(f"[DEBUG] Finalize activated {self.state.debug()}")

    @kernel_function
    async def summarize(self, ctx: KernelProcessStepContext, state: FighterState):
        self.state = state
        print("\n" + "=" * 40)
        print("TOURNAMENT FLOW COMPLETE")
        print("=" * 40)
        print(f"Fighter        : {self.state.fighter_name}")
        print(f"Approved gear  : {self.state.approved}")
        print(f"Injury occurred: {self.state.injury_occurred}")
        print("=" * 40 + "\n")
        journey_lines = "\n".join(f"- {item}" for item in self.state.journey)
        summary = (
            f"**Tournament summary**\n"
            f"- Fighter : {self.state.fighter_name}\n"
            f"- Gear OK : {self.state.approved}\n"
            f"- Injury  : {self.state.injury_occurred}"
            f"\n**Journey:**\n{journey_lines}"
        )
        self.state.summary = summary
        # self.state.say(summary)

        await ctx.emit_event(process_event=FighterEvents.ProcessComplete, data=None)

# ─────────────────────────────────────────────────────────────────────────────
# KERNEL  (kept minimal; only needed if you still want Azure-OpenAI access)
def create_kernel(service_id: str = "process-framework") -> Kernel:
    kernel = Kernel()
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        kernel.add_service(AzureChatCompletion(service_id=service_id))
    return kernel

# ─────────────────────────────────────────────────────────────────────────────
async def run_tournament_flow(notify):
    kernel = create_kernel()            # optional; not used in steps above
    process = ProcessBuilder(name="TournamentFlow")

    # build steps
    reg   = process.add_step(RegistrationStep)
    lines = process.add_step(LinesStep)
    pen   = process.add_step(BullpenStep)
    fix   = process.add_step(ResolveIssuesStep)
    wait  = process.add_step(WaitingStep)
    ring  = process.add_step(FightStep)
    med   = process.add_step(MedicStep)
    fin   = process.add_step(FinalizeStep)

    # wiring  (matches the Mermaid diagram)
    process.on_input_event(FighterEvents.StartProcess).send_event_to(reg, parameter_name="state")

    reg  .on_event(FighterEvents.RegistrationDone).send_event_to(lines, parameter_name="state")
    lines.on_event(FighterEvents.LinesCleared    ).send_event_to(pen, parameter_name="state")

    pen  .on_event(FighterEvents.Approved ).send_event_to(wait, parameter_name="state")
    pen  .on_event(FighterEvents.Rejected ).send_event_to(fix, parameter_name="state")

    fix  .on_event(FighterEvents.GearIssuesResolved).send_event_to(pen, parameter_name="state")

    wait .on_event(FighterEvents.WaitingOver).send_event_to(ring, parameter_name="state")

    ring .on_event(FighterEvents.Injury   ).send_event_to(med, parameter_name="state")
    ring .on_event(FighterEvents.FightOver).send_event_to(fin, parameter_name="state")
    med  .on_event(FighterEvents.FightOver).send_event_to(fin, parameter_name="state")

    fin  .on_event(FighterEvents.ProcessComplete).stop_process()

    # initial state object that carries notify
    initial_state = FighterState(notify=notify)

    # run
    await start(
        process   = process.build(),
        kernel    = kernel,
        initial_event=KernelProcessEvent(
            id=FighterEvents.StartProcess,
            data=initial_state               # ← here
        ),
    )
    return f"Tournament summary is ready.\n{initial_state.summary}"

# if __name__ == "__main__":
#     asyncio.run(run_tournament_flow())
