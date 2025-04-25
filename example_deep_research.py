import asyncio
import os
from minion_agent.config import AgentConfig
from minion_agent.frameworks.deep_research import DeepResearchAgent

async def main():
    # Configure the deep research agent
    config = AgentConfig(
        name="Deep Research Assistant",

        tools=[
        ]
    )
    main_agent_config = AgentConfig(
        model_id=os.environ.get("AZURE_DEPLOYMENT_NAME"),
        name="research_assistant",
        description="A helpful research assistant",
        model_args={"azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
                    "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
                    "api_version": os.environ.get("OPENAI_API_VERSION"),
                    },
        tools=[
            # "minion_agent.tools.browser_tool.browser",
            MCPTool(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "/Users/femtozheng/workspace",
                      "/Users/femtozheng/python-project/minion-agent"]
            )
        ],

    # Initialize the deep research agent
    research_agent = DeepResearchAgent(
        config=config,
        together_api_key=os.getenv("TOGETHER_API_KEY")
    )

    # Example research query
    research_query = """
    Research Topic: The current state and future potential of quantum computing
    
    Please investigate:
    1. Recent breakthroughs in quantum computing technology
    2. Major players and their approaches
    3. Current limitations and challenges
    4. Potential applications across industries
    5. Timeline predictions for practical quantum computers
    
    Focus on verified sources and provide a comprehensive analysis.
    """

    try:
        # Run the research
        result = await research_agent.run_async(research_query)
        
        print("\n=== Research Results ===\n")
        print(result)
        
    except Exception as e:
        print(f"Error during research: {str(e)}")

if __name__ == "__main__":
    # Set up environment variables if needed
    if not os.getenv("TOGETHER_API_KEY"):
        print("Please set TOGETHER_API_KEY environment variable")
        exit(1)
        
    asyncio.run(main()) 