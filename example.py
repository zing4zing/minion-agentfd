"""Example usage of Minion Agent."""

import asyncio
from dotenv import load_dotenv
import os
from PIL import Image
from io import BytesIO
from time import sleep

from minion_agent.config import MCPTool

# Load environment variables from .env file
load_dotenv()

from minion_agent import MinionAgent, AgentConfig, AgentFramework

from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
    HfApiModel, AzureOpenAIServerModel, ActionStep,
)

# Set up screenshot callback for Playwright
def save_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0)  # Let JavaScript animations happen before taking the screenshot
    
    # Get the browser tool
    browser_tool = agent.tools.get("browser")
    if browser_tool:
        # Clean up old screenshots to save memory
        for previous_memory_step in agent.memory.steps:
            if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= memory_step.step_number - 2:
                previous_memory_step.observations_images = None
        
        # Take screenshot using Playwright
        result = browser_tool(action="screenshot")
        if result["success"] and "screenshot" in result.get("data", {}):
            # Convert bytes to PIL Image
            screenshot_bytes = result["data"]["screenshot"]
            image = Image.open(BytesIO(screenshot_bytes))
            print(f"Captured a browser screenshot: {image.size} pixels")
            memory_step.observations_images = [image.copy()]  # Create a copy to ensure it persists
        
        # Get current URL
        state_result = browser_tool(action="get_current_state")
        if state_result["success"] and "url" in state_result.get("data", {}):
            url_info = f"Current url: {state_result['data']['url']}"
            memory_step.observations = (
                url_info if memory_step.observations is None else memory_step.observations + "\n" + url_info
            )

# Configure the agent
agent_config = AgentConfig(
    model_id=os.environ.get("AZURE_DEPLOYMENT_NAME"),  # 使用你的API部署中可用的模型
    name="research_assistant",
    description="A helpful research assistant",
    model_args={"azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
                "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
                "api_version": os.environ.get("OPENAI_API_VERSION"),
                },
    tools=[
        "minion_agent.tools.browser_tool.browser",
        MCPTool(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem","/Users/femtozheng/workspace"]
        )
    ],
    agent_type="CodeAgent",
    model_type="AzureOpenAIServerModel",
    agent_args={"additional_authorized_imports":"*",
                #"step_callbacks":[save_screenshot]
                }
)

# from opentelemetry.sdk.trace import TracerProvider
#
# from openinference.instrumentation.smolagents import SmolagentsInstrumentor
# from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
# from opentelemetry.sdk.trace.export import SimpleSpanProcessor
#
# otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
# trace_provider = TracerProvider()
# trace_provider.add_span_processor(SimpleSpanProcessor(otlp_exporter))
#
# SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

async def main():
    try:
        # Create and run the agent
        agent = await MinionAgent.create(AgentFramework.SMOLAGENTS, agent_config)
        
        # Run the agent with a question
        #result = await agent.run("search sam altman and export summary as markdown")
        #result = await agent.run("What are the latest developments in AI, find this information and export as markdown")
        result = await agent.run("搜索最新的人工智能发展趋势，并且总结为markdown")
        #result = await agent.run("go visit https://www.baidu.com and clone it")
        #result = await agent.run("go visit https://www.baidu.com , take a screenshot and clone it")
        #result = await agent.run("实现一个贪吃蛇游戏")
        print("Agent's response:", result)
    except Exception as e:
        print(f"Error: {str(e)}")
        # 如果需要调试
        # import litellm
        # litellm._turn_on_debug()
        raise

if __name__ == "__main__":
    asyncio.run(main()) 