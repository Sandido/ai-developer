# tournament_flow.py
import os, asyncio, logging, random
from enum import Enum
from dotenv import load_dotenv

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

# ─────────────────────────────────────────────────────────────────────────────
# logging / env
logger = logging.getLogger(__name__)
load_dotenv(override=True)

# ─────────────────────────────────────────────────────────────────────────────
# EVENTS ─ mirrors the Mermaid edges
class FighterEvents(str, Enum):
    StartProcess     = "startProcess"
    RegistrationDone = "registrationDone"
    LinesCleared     = "linesCleared"
    Approved         = "approved"
    Rejected         = "rejected"
    IssuesResolved   = "issuesResolved"
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

    def debug(self) -> str:
        return f"(name={self.fighter_name!r}, approved={self.approved}, injury={self.injury})"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1  – Registration
class RegistrationStep(KernelProcessStep[FighterState]):
    async def activate(self, step_state: KernelProcessStepState[FighterState]):
        step_state.state = step_state.state or FighterState()
        self.state = step_state.state
        print(f"[DEBUG] Registration activated {self.state.debug()}")

    @kernel_function
    async def register(self, ctx: KernelProcessStepContext):
        self.state.fighter_name = input("Fighter name: ").strip()
        print("✔  Registration complete.")
        await ctx.emit_event(process_event=FighterEvents.RegistrationDone, data=self.state)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Lines
class LinesStep(KernelProcessStep[FighterState]):
    async def activate(self, st: KernelProcessStepState[FighterState]):
        self.state = st.state or FighterState()
        print(f"[DEBUG] Lines activated {self.state.debug()}")

    @kernel_function
    async def wait_in_line(self, ctx: KernelProcessStepContext):
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
    ):
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
            await ctx.emit_event(process_event=FighterEvents.Approved, data=self.state)
        elif verdict.startswith("reject"):
            self.state.approved = False
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
    async def fix_gear(self, ctx: KernelProcessStepContext):
        print(f"⚒  Resolving: {self.state.gear_notes}")
        input("Press ENTER when issues are fixed and fighter returns to Bullpen… ")
        await ctx.emit_event(process_event=FighterEvents.IssuesResolved, data=self.state)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Pre-Fight Waiting Area
class WaitingStep(KernelProcessStep[FighterState]):
    async def activate(self, st: KernelProcessStepState[FighterState]):
        self.state = st.state or FighterState()
        print(f"[DEBUG] WaitingArea activated {self.state.debug()}")

    @kernel_function
    async def wait(self, ctx: KernelProcessStepContext):
        input("Waiting… (ENTER when fighter is called to ring) ")
        await ctx.emit_event(process_event=FighterEvents.WaitingOver, data=self.state)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 – Fight Rings
class FightStep(KernelProcessStep[FighterState]):
    async def activate(self, st: KernelProcessStepState[FighterState]):
        self.state = st.state or FighterState()
        print(f"[DEBUG] Fight activated {self.state.debug()}")

    @kernel_function
    async def fight(self, ctx: KernelProcessStepContext):
        # 1 % injury chance
        self.state.injury = random.random() < 0.5 #0.01
        if self.state.injury:
            print("⚠  Fighter injured! Sending to medic.")
            await ctx.emit_event(process_event=FighterEvents.Injury, data=self.state)
        else:
            print("✔  Fight completed without injury.")
            await ctx.emit_event(process_event=FighterEvents.FightOver, data=self.state)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 – Medic
class MedicStep(KernelProcessStep[FighterState]):
    async def activate(self, st: KernelProcessStepState[FighterState]):
        self.state = st.state or FighterState()
        print(f"[DEBUG] Medic activated {self.state.debug()}")

    @kernel_function(description="To treat the fighter after injury to make them recover.")
    async def treat(self, ctx: KernelProcessStepContext):
        input("Medic treatment done (press ENTER)… ")
        await ctx.emit_event(process_event=FighterEvents.FightOver, data=self.state)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 – Finalize / End
class FinalizeStep(KernelProcessStep[FighterState]):
    async def activate(self, st: KernelProcessStepState[FighterState]):
        self.state = st.state or FighterState()
        print(f"[DEBUG] Finalize activated {self.state.debug()}")

    @kernel_function
    async def summarize(self, ctx: KernelProcessStepContext):
        print("\n" + "=" * 40)
        print("TOURNAMENT FLOW COMPLETE")
        print("=" * 40)
        print(f"Fighter        : {self.state.fighter_name}")
        print(f"Approved gear  : {self.state.approved}")
        print(f"Injury occurred: {self.state.injury}")
        print("=" * 40 + "\n")
        await ctx.emit_event(process_event=FighterEvents.ProcessComplete, data=None)

# ─────────────────────────────────────────────────────────────────────────────
# KERNEL  (kept minimal; only needed if you still want Azure-OpenAI access)
def create_kernel(service_id: str = "process-framework") -> Kernel:
    kernel = Kernel()
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        kernel.add_service(AzureChatCompletion(service_id=service_id))
    return kernel

# ─────────────────────────────────────────────────────────────────────────────
async def run_tournament_flow():
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
    process.on_input_event(FighterEvents.StartProcess).send_event_to(reg)

    reg  .on_event(FighterEvents.RegistrationDone).send_event_to(lines)
    lines.on_event(FighterEvents.LinesCleared    ).send_event_to(pen)

    pen  .on_event(FighterEvents.Approved ).send_event_to(wait)
    pen  .on_event(FighterEvents.Rejected ).send_event_to(fix)

    fix  .on_event(FighterEvents.IssuesResolved).send_event_to(pen)

    wait .on_event(FighterEvents.WaitingOver).send_event_to(ring)

    ring .on_event(FighterEvents.Injury   ).send_event_to(med)
    ring .on_event(FighterEvents.FightOver).send_event_to(fin)
    med  .on_event(FighterEvents.FightOver).send_event_to(fin)

    fin  .on_event(FighterEvents.ProcessComplete).stop_process()

    # run
    await start(
        process   = process.build(),
        kernel    = kernel,
        initial_event = KernelProcessEvent(id=FighterEvents.StartProcess, data=None),
    )

if __name__ == "__main__":
    asyncio.run(run_tournament_flow())
