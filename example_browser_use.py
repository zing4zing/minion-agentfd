import os
import asyncio
from dotenv import load_dotenv
from minion_agent.config import AgentConfig
from minion_agent.frameworks.browser_use import BrowserUseAgent

# Load environment variables from .env file
load_dotenv()

async def main():
    # Create agent configuration
    config = AgentConfig(
        name="browser-agent",
        model_id="gpt-4.1",  # Your Azure OpenAI deployed model name
        model_args={
            "api_version": "2024-02-15",
            "temperature": 0
        },
        # You can specify initial instructions here
        instructions="Compare the price of gpt-4o and DeepSeek-V3"
    )

    # Initialize the browser use agent
    agent = BrowserUseAgent(config)
    
    # Make sure the agent is loaded before running
    if not agent._agent_loaded:
        await agent._load_agent()

    # Run the agent with a specific task
    result = await agent.run_async("Compare the price of gpt-4o and DeepSeek-V3 and create a detailed comparison table")
    print("Task Result:", result)

    # Run another task
    result = await agent.run_async("Go to baidu.com, search for '人工智能最新进展', and summarize the top 3 results")
    print("Task Result:", result)

if __name__ == "__main__":
    # Verify environment variables
    required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("Error: Missing required environment variables:", missing_vars)
        print("Please set these variables in your .env file:")
        print("""
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_KEY=your_key_here
        """)
    else:
        asyncio.run(main()) 