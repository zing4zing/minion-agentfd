import os
import importlib
from typing import Optional, Any, List
from pydantic import SecretStr

from minion_agent.config import AgentFramework, AgentConfig
from minion_agent.frameworks.minion_agent import MinionAgent
from minion_agent.tools.wrappers import import_and_wrap_tools

try:
    from browser_use import Agent
    browser_use_available = True
except ImportError:
    browser_use_available = None

DEFAULT_MODEL_CLASS = "langchain_openai.ChatOpenAI"

class BrowserUseAgent(MinionAgent):
    """Browser-use agent implementation that handles both loading and running."""
    name = "browser_agent"
    description = "Browser-use agent"

    def __init__(
        self, config: AgentConfig, managed_agents: Optional[list[AgentConfig]] = None
    ):
        if not browser_use_available:
            raise ImportError(
                "You need to `pip install 'minion-agent[browser_use]'` to use this agent"
            )
        self.managed_agents = managed_agents
        self.config = config
        self._agent = None
        self._agent_loaded = False
        self._mcp_servers = None

    def _get_model(self, agent_config: AgentConfig):
        """Get the model configuration for a LangChain agent."""
        if not agent_config.model_type:
            agent_config.model_type = DEFAULT_MODEL_CLASS
        module, class_name = agent_config.model_type.split(".")
        model_type = getattr(importlib.import_module(module), class_name)

        return model_type(model=agent_config.model_id, **agent_config.model_args or {})

    async def _load_agent(self) -> None:
        """Load the Browser-use agent with the given configuration."""
        if not self.config.tools:
            self.config.tools = []  # Browser-use has built-in browser automation tools

        tools, mcp_servers = await import_and_wrap_tools(
            self.config.tools, agent_framework=AgentFramework.BROWSER_USE
        )
        self._mcp_servers = mcp_servers

        # Initialize the browser-use Agent
        self._agent = Agent(
            task=self.config.instructions or "No specific task provided",
            llm=self._get_model(self.config)
        )

    async def run_async(self, prompt: str) -> Any:
        """Run the Browser-use agent with the given prompt."""
        # Update the agent's task with the new prompt
        self._agent.task = prompt
        result = await self._agent.run()
        return result

    @property
    def tools(self) -> List[str]:
        """
        Return the tools used by the agent.
        This property is read-only and cannot be modified.
        """
        return ["browser_automation"]  # Browser-use has built-in browser tools
