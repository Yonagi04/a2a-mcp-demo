import asyncio
import uvicorn
import os

from a2a.types import AgentSkill, AgentCard, AgentCapabilities
import asyncclick as click
from a2a.server.request_handlers import DefaultRequestHandler

from agents.arithmetic_agent.agent_executor import ArithmeticAgentExecutor
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.apps import A2AStarletteApplication

@click.command()
@click.option('--host', default='localhost', help='Host for the agent server')
@click.option('--port', default=10002, help='Port for the agent server')
async def main(host: str, port: int):
    """
    Main function to create and run the arithmetic agent.
    """
    skill = AgentSkill(
        id="arithmetic_agent_skill",
        name="arithmetic_agent_skill",
        description="A simple arithmetic agent that can caculate",
        tags=["arithmetic", "agent", "add", "subtract", "multiply", "divide"],
        examples=[
            """Caculate: 1023+271=?""",
            """2817*192=?""",
            """182/21=?"""
        ]
    )

    agent_card = AgentCard(
        name ="arithmetic_agent",
        description="A simple arithmetic agent that can caculate",
        url=os.getenv("ARITHMETIC_AGENT_URL", f"http://{host}:{port}/"),
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[skill],
        capabilities=AgentCapabilities(streaming=True),
    )

    # Create agent executor
    agent_executor = ArithmeticAgentExecutor()
    await agent_executor.create()

    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore()
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )

    # Fixed: Use uvicorn.Config and Server instead of uvicorn.run() to avoid
    # "asyncio.run() cannot be called from a running event loop" error
    config = uvicorn.Config(server.build(), host=host, port=port)
    server_instance = uvicorn.Server(config)

    await server_instance.serve()

if __name__ == "__main__":
    asyncio.run(main())