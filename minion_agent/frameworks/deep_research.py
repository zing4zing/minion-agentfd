import os
from typing import Optional, Any, List, Dict
from dataclasses import fields

from together import Together
from langgraph.graph import Graph, StateGraph
from langchain import PromptTemplate
from langchain_together import TogetherChat
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent

from minion_agent.config import AgentFramework, AgentConfig
from minion_agent.frameworks.minion_agent import MinionAgent
from minion_agent.instructions import get_instructions
from minion_agent.logging import logger
from minion_agent.tools.wrappers import import_and_wrap_tools

RESEARCH_MAX_ITERATIONS = 5

class DeepResearchAgent(MinionAgent):
    """Deep research agent implementation using Together AI and LangGraph."""

    def __init__(
        self, 
        config: AgentConfig, 
        managed_agents: Optional[list[AgentConfig]] = None,

    ):
        self.managed_agents = managed_agents
        self.config = config
        self._agent = None
        self._agent_loaded = False
        self.together_api_key = os.getenv("TOGETHER_API_KEY")
        if not self.together_api_key:
            raise ValueError("TOGETHER_API_KEY must be provided or set in environment")
        
        Together.api_key = self.together_api_key

    async def _setup_research_tools(self) -> List[Tool]:
        """Set up the core research tools."""
        tools = []
        
        # Import configured tools
        if not self.config.tools:
            self.config.tools = [
                "minion_agent.tools.search_web",
                "minion_agent.tools.visit_webpage",
                "minion_agent.tools.analyze_document",
                "minion_agent.tools.extract_content",
                "minion_agent.tools.summarize_text"
            ]
        
        imported_tools, mcp_servers = await import_and_wrap_tools(
            self.config.tools,
            agent_framework=AgentFramework.TOGETHER
        )
        tools.extend(imported_tools)
        
        return tools, mcp_servers

    def _create_research_nodes(self) -> Dict:
        """Create the research workflow nodes."""
        nodes = {
            "search": self._search_node,
            "analyze": self._analyze_node,
            "synthesize": self._synthesize_node,
            "validate": self._validate_node,
            "decide": self._decide_node
        }
        return nodes

    async def _search_node(self, state):
        """Search for relevant information."""
        # Implementation for search node
        pass

    async def _analyze_node(self, state):
        """Analyze gathered information."""
        # Implementation for analyze node
        pass

    async def _synthesize_node(self, state):
        """Synthesize findings into coherent results."""
        # Implementation for synthesize node
        pass

    async def _validate_node(self, state):
        """Validate findings and check for gaps."""
        # Implementation for validate node
        pass

    async def _decide_node(self, state):
        """Decide next steps or conclude research."""
        # Implementation for decide node
        pass

    async def _load_agent(self) -> None:
        """Load the deep research agent with LangGraph workflow."""
        # Set up base tools
        tools, mcp_servers = await self._setup_research_tools()

        # Create research workflow
        workflow = StateGraph(nodes=self._create_research_nodes())
        
        # Define workflow edges
        workflow.add_edge('search', 'analyze')
        workflow.add_edge('analyze', 'synthesize')
        workflow.add_edge('synthesize', 'validate')
        workflow.add_edge('validate', 'decide')
        workflow.add_edge('decide', 'search')  # Loop back if needed
        
        # Set up the model
        model = TogetherChat(
            model=self.config.model_id or "mistral-7b-instruct",
            together_api_key=self.together_api_key,
            temperature=0.7,
            max_tokens=2048
        )

        # Create the agent
        prompt = PromptTemplate.from_template(
            get_instructions(self.config.instructions)
        )
        
        self._agent = create_react_agent(
            llm=model,
            tools=tools,
            prompt=prompt
        )
        
        self._executor = AgentExecutor(
            agent=self._agent,
            tools=tools,
            max_iterations=RESEARCH_MAX_ITERATIONS,
            early_stopping_method="generate"
        )

        self._workflow = workflow
        self._agent_loaded = True

    async def run_async(self, prompt: str) -> Any:
        """Run the deep research workflow asynchronously."""
        if not self._agent_loaded:
            await self._load_agent()

        # Initialize research state
        initial_state = {
            "query": prompt,
            "findings": [],
            "current_step": "search",
            "iterations": 0,
            "complete": False
        }

        # Run the research workflow
        try:
            result = await self._workflow.arun(
                initial_state,
                max_iterations=RESEARCH_MAX_ITERATIONS
            )
            
            # Process final results through the agent
            final_response = await self._executor.arun(
                input=f"Synthesize the following research findings into a comprehensive response: {result['findings']}"
            )
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error during research workflow: {str(e)}")
            raise

    @property
    def tools(self) -> List[str]:
        """Return the tools used by the agent."""
        if not hasattr(self, '_agent'):
            logger.warning("Agent not loaded or does not have tools.")
            return []
            
        tools = [tool.name for tool in self._agent.tools]
        return tools 