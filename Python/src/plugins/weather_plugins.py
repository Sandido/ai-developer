# TODO: placeholder plugin for now. This is another external API plugin.

from semantic_kernel import Kernel
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