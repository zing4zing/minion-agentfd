import os
from typing import Optional, Any, List

from minion_agent.config import AgentFramework, AgentConfig
from minion_agent.frameworks.minion_agent import MinionAgent
from minion_agent.tools.wrappers import import_and_wrap_tools

try:
    import minion
    from minion.main.brain import Brain
    from minion.providers import create_llm_provider
    from minion import config as minion_config
    from minion.main.local_python_env import LocalPythonEnv
    minion_available = True
except ImportError as e:
    minion_available = None



class MinionBrainAgent(MinionAgent):
    """minion agent implementation that handles both loading and running."""

    def __init__(
        self, config: AgentConfig, managed_agents: Optional[list[AgentConfig]] = None
    ):
        if not minion_available:
            raise ImportError(
                "You need to `pip install 'minion-agent-x[minion]'` to use this agent"
            )
        self.managed_agents = managed_agents
        self.config = config
        self._agent = None
        self._agent_loaded = False
        self._mcp_servers = None
        self._managed_mcp_servers = None

    def _get_model(self, agent_config: AgentConfig):
        """Get the model configuration for a minion agent.
        
        Args:
            agent_config: The agent configuration containing model settings
            
        Returns:
            A minion provider instance configured with the specified model
        """
        # Get model ID from config or use default
        model_id = agent_config.model_id or "gpt-4o"
        
        # Get model config from minion's config
        llm_config = minion_config.models.get(model_id)
        if not llm_config:
            raise ValueError(f"Model {model_id} not found in minion config")
            
        # Create provider with model args from agent config
        provider = create_llm_provider(
            llm_config
        )
        
        return provider

    def _merge_mcp_tools(self, mcp_servers):
        """Merge MCP tools from different servers."""
        tools = []
        for mcp_server in mcp_servers:
            tools.extend(mcp_server.tools)
        return tools

    async def _load_agent(self) -> None:
        """Load the agent instance with the given configuration."""
        if not self.managed_agents and not self.config.tools:
            self.config.tools = [
                "minion_agent.tools.search_web",
                "minion_agent.tools.visit_webpage",
            ]

        tools, mcp_servers = await import_and_wrap_tools(
            self.config.tools, agent_framework=AgentFramework.SMOLAGENTS
        )
        self._mcp_servers = mcp_servers
        tools.extend(self._merge_mcp_tools(mcp_servers))

        managed_agents_instanced = []
        if self.managed_agents:
            for managed_agent in self.managed_agents:
                agent_type = getattr(
                    smolagents, managed_agent.agent_type or DEFAULT_AGENT_TYPE
                )
                managed_tools, managed_mcp_servers = await import_and_wrap_tools(
                    managed_agent.tools, agent_framework=AgentFramework.SMOLAGENTS
                )
                self._managed_mcp_servers = managed_mcp_servers
                tools.extend(self._merge_mcp_tools(managed_mcp_servers))
                managed_agent_instance = agent_type(
                    name=managed_agent.name,
                    model=self._get_model(managed_agent),
                    tools=managed_tools,
                    verbosity_level=2,  # OFF
                    description=managed_agent.description
                    or f"Use the agent: {managed_agent.name}",
                )
                if managed_agent.instructions:
                    managed_agent_instance.prompt_templates["system_prompt"] = (
                        managed_agent.instructions
                    )
                managed_agents_instanced.append(managed_agent_instance)

        main_agent_type = Brain

        # Get python_env from config or use default
        agent_args = self.config.agent_args or {}
        python_env = agent_args.pop('python_env', None) or LocalPythonEnv(verbose=False)

        self._agent = main_agent_type(
            python_env=python_env,
            llm=self._get_model(self.config),
            **agent_args
        )

    async def run_async(self, task: str,*args,**kwargs) -> Any:
        """Run the Smolagents agent with the given prompt."""
        obs, *_ = await self._agent.step(query=task, *args, **kwargs)
        return obs

    @property
    def tools(self) -> List[str]:
        """
        Return the tools used by the agent.
        This property is read-only and cannot be modified.
        """
        return self._agent.tools
