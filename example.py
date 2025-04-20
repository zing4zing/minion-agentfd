"""Example usage of Minion Agent."""

import asyncio
from dotenv import load_dotenv
import os

from minion_agent.config import MCPTool

# Load environment variables from .env file
load_dotenv()

from minion_agent import MinionAgent, AgentConfig, AgentFramework

from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
    HfApiModel, AzureOpenAIServerModel,
)

# 配置 Azure OpenAI 模型
# model = AzureOpenAIServerModel(
#     model_id=os.environ.get("AZURE_OPENAI_MODEL"),          # 例如 "gpt-4" 或 "gpt-35-turbo"
#     azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),  # Azure OpenAI 服务端点
#     api_key=os.environ.get("AZURE_OPENAI_API_KEY"),         # Azure OpenAI API 密钥
#     api_version=os.environ.get("OPENAI_API_VERSION")        # API 版本，例如 "2024-02-15-preview"
# )
# Configure the agent
agent_config = AgentConfig(
    model_id=os.environ.get("AZURE_DEPLOYMENT_NAME"),  # 使用你的API部署中可用的模型
    name="research_assistant",
    description="A helpful research assistant",
    # model_args={"api_key_var": "OPENAI_API_KEY", "base_url_var":"OPENAI_BASE_URL"},  # Will use OPENAI_API_KEY from environment
    model_args={"azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
                "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
                "api_version": os.environ.get("OPENAI_API_VERSION"),
                },  # Will use OPENAI_API_KEY from environment
    #tools=["minion_agent.tools.web_browsing.search_web"],
    tools=[
        #"minion_agent.tools.web_browsing.search_web","minion_agent.tools.web_browsing.visit_webpage",
        "minion_agent.tools.browser_tool.browser",
        MCPTool(
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-filesystem","/Users/femtozheng/workspace"]

                )
           ],
    agent_type="CodeAgent",
    model_type="AzureOpenAIServerModel",
    #agent_args={"additional_authorized_imports":["openai","bs4","os"]}
    agent_args={"additional_authorized_imports":"*"}
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
        
        result = await agent.run("go visit https://www.baidu.com and clone it")
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