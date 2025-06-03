# TODO: placeholder plugin for now

from semantic_kernel import Kernel
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.functions import kernel_function
import logging

# Add Logger
logger = logging.getLogger(__name__)

class WeatherPlugin:
    """A plugin that returns the weather."""
    def __init__(self, kernel: Kernel):
        logger.info("Weather Plugin init !!!!!!!!!!!!")
        
        
    @kernel_function(description="Answers questions about the weather.")
    def get_weather(self) -> str:
        """Retrieve the weather."""
        logger.info("Weather Plugin get_weather !!!!!!!!!!!!")
        return f"It's always sunny in Philadelphia!"    