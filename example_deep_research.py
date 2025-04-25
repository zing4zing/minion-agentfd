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
        name="main_agent",
        description="main agent",
        model_args={"azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
                    "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
                    "api_version": os.environ.get("OPENAI_API_VERSION"),
                    },
        tools=[
        ],
    )
    research_agent_config = AgentConfig(
        framework=AgentFramework.DEEP_RESEARCH,
        model_id=os.environ.get("AZURE_DEPLOYMENT_NAME"),
        name="research_assistant",
        description="A helpful research assistant",
        model_args={"azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
                    "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
                    "api_version": os.environ.get("OPENAI_API_VERSION"),
                    },
        tools=[
        ],
    )
    main_agent = MinionAgent.create(
        AgentFramework.SMOLAGENTS,
        managed_agents=[research_agent_config],
    )

    # Example research query
    research_query = """
    The evolution of Indo-European languages
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