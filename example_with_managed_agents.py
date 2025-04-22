import os
import asyncio
from dotenv import load_dotenv
from minion_agent import MinionAgent, AgentConfig, AgentFramework
from minion_agent.config import MCPTool

# Load environment variables from .env file
load_dotenv()

async def main():
    # Get Azure configuration from environment variables
    azure_deployment = os.getenv('AZURE_DEPLOYMENT_NAME')
    api_version = os.getenv('OPENAI_API_VERSION')
    if not azure_deployment:
        raise ValueError("AZURE_DEPLOYMENT_NAME environment variable is not set")
    if not api_version:
        raise ValueError("OPENAI_API_VERSION environment variable is not set")

    # Create main agent configuration with MCP filesystem tool
    main_agent_config = AgentConfig(
    model_id=os.environ.get("AZURE_DEPLOYMENT_NAME"),
    name="research_assistant",
    description="A helpful research assistant",
    model_args={"azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
                "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
                "api_version": os.environ.get("OPENAI_API_VERSION"),
                },
    tools=[
        #"minion_agent.tools.browser_tool.browser",
        MCPTool(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem","/Users/femtozheng/workspace"]
        )
    ],
    agent_type="CodeAgent",
    model_type="AzureOpenAIServerModel",  # Updated to use our custom model
    #model_type="CustomAzureOpenAIServerModel",  # Updated to use our custom model
    agent_args={"additional_authorized_imports":"*",
                #"planning_interval":3
                }
)

    # Create browser agent configuration
    browser_agent_config = AgentConfig(
        name="browser_agent",
        model_type="langchain_openai.AzureChatOpenAI",
        model_id=azure_deployment,
        model_args={
            "azure_deployment": azure_deployment,
            "api_version": api_version,
        },
        tools=[],
        instructions="I am a browser agent that can perform web browsing tasks."
    )
    browser_agent = await MinionAgent.create(
        AgentFramework.BROWSER_USE,
        browser_agent_config,
    )

    # Create and initialize the main agent with the browser agent as managed agent
    agent = await MinionAgent.create(
        AgentFramework.SMOLAGENTS,
        main_agent_config,
        managed_agents=[browser_agent]
    )

    # Example tasks that combine filesystem and browser capabilities
    tasks = [
        "Search for 'latest AI developments' and save the results to a markdown file",
        "Visit baidu.com, take a screenshot, and save it to the workspace",
        "Compare GPT-4 and Claude pricing, create a comparison table, and save it as a markdown document"
    ]

    for task in tasks:
        print(f"\nExecuting task: {task}")
        result = await agent.run_async(task)
        print("Task Result:", result)

if __name__ == "__main__":
    # Verify environment variables
    required_vars = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_API_KEY',
        'AZURE_DEPLOYMENT_NAME',
        'OPENAI_API_VERSION'
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("Error: Missing required environment variables:", missing_vars)
        print("Please set these variables in your .env file:")
        print("""
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_API_KEY=your_key_here
AZURE_DEPLOYMENT_NAME=your_deployment_name_here
OPENAI_API_VERSION=your_api_version_here  # e.g. 2024-02-15
        """)
    else:
        asyncio.run(main()) 