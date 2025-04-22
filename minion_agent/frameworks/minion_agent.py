from typing import Any, Optional, List
from abc import ABC, abstractmethod
import asyncio
from minion_agent.config import AgentFramework, AgentConfig


class MinionAgent(ABC):
    """Base abstract class for all agent implementations.

    This provides a unified interface for different agent frameworks.
    """

    # factory method
    @classmethod
    async def create(
        cls,
        agent_framework: AgentFramework,
        agent_config: AgentConfig,
        managed_agents: Optional[list[AgentConfig]] = None,
    ) -> "MinionAgent":
        if agent_framework == AgentFramework.SMOLAGENTS:
            from minion_agent.frameworks.smolagents import SmolagentsAgent as Agent
        elif agent_framework == AgentFramework.LANGCHAIN:
            from minion_agent.frameworks.langchain import LangchainAgent as Agent
        elif agent_framework == AgentFramework.OPENAI:
            from minion_agent.frameworks.openai import OpenAIAgent as Agent
        elif agent_framework == AgentFramework.LLAMAINDEX:
            from minion_agent.frameworks.llama_index import LlamaIndexAgent as Agent
        elif agent_framework == AgentFramework.GOOGLE:
            from minion_agent.frameworks.google import GoogleAgent as Agent
        elif agent_framework == AgentFramework.MINION:
            from minion_agent.frameworks.minion import MinionBrainAgent as Agent
        else:
            raise ValueError(f"Unsupported agent framework: {agent_framework}")
            
        agent = Agent(agent_config, managed_agents=managed_agents)
        await agent._load_agent()
        return agent

    @abstractmethod
    async def _load_agent(self) -> None:
        """Load the agent instance."""
        pass

    async def run(self, prompt: str) -> Any:
        """Run the agent with the given prompt."""
        return await self.run_async(prompt)

    @abstractmethod
    async def run_async(self, prompt: str) -> Any:
        """Run the agent asynchronously with the given prompt."""
        pass

    @property
    @abstractmethod
    def tools(self) -> List[str]:
        """
        Return the tools used by the agent.
        This property is read-only and cannot be modified.
        """
        pass

    def __init__(self):
        raise NotImplementedError(
            "Cannot instantiate the base class AnyAgent, please use the factory method 'AnyAgent.create'"
        )

    @property
    def agent(self):
        """
        The underlying agent implementation from the framework.

        This property is intentionally restricted to maintain framework abstraction
        and prevent direct dependency on specific agent implementations.

        If you need functionality that relies on accessing the underlying agent:
        1. Consider if the functionality can be added to the AnyAgent interface
        2. Submit a GitHub issue describing your use case
        3. Contribute a PR implementing the needed functionality

        Raises:
            NotImplementedError: Always raised when this property is accessed
        """
        raise NotImplementedError(
            "Cannot access the 'agent' property of AnyAgent, if you need to use functionality that relies on the underlying agent framework, please file a Github Issue or we welcome a PR to add the functionality to the AnyAgent class"
        )
