"""Minion Manus - A wrapper for smolagents."""

__version__ = "0.1.0"


from .config import AgentConfig, AgentFramework, Settings
from .frameworks.minion_agent import MinionAgent
from .utils import setup_logging

settings = Settings.from_env()  # 或传入自定义设置
setup_logging(settings)

__all__ = ["MinionAgent", "AgentConfig", "AgentFramework"]
