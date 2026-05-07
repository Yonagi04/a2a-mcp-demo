from utilities.mcp.mcp_connect import MCPConnector

from collections.abc import AsyncIterable
from utilities.common.file_loader import load_instructions_file
from google.adk.agents import LlmAgent
from google.adk import Runner
from google.adk.models.lite_llm import LiteLlm

from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService

from google.genai import types

from rich import print as rprint
from rich.syntax import Syntax

import json
import os
import time
from typing import Any
from utilities.common.network_logger import network_logger

from dotenv import load_dotenv
load_dotenv()

async def llm_before_model_callback(callback_context, llm_request):
    """LLM 调用开始前的回调，记录开始时间"""
    callback_context.state["_llm_call_start"] = time.time()
    callback_context.state["_llm_model"] = getattr(llm_request, 'model', 'unknown')
    return None


async def llm_after_model_callback(callback_context, llm_response):
    """LLM 调用完成后的回调，记录调用时长"""
    start_time = callback_context.state.get("_llm_call_start")
    model = callback_context.state.get("_llm_model", "unknown")

    if start_time:
        latency = time.time() - start_time
        source = callback_context.agent_name if hasattr(callback_context, 'agent_name') else "unknown"

        # 获取 token 使用量
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(llm_response, 'usage_metadata') and llm_response.usage_metadata:
            usage = llm_response.usage_metadata
            prompt_tokens = getattr(usage, 'prompt_token_count', 0)
            completion_tokens = getattr(usage, 'candidates_token_count', 0)

        # 判断响应状态
        response_status = "success"
        error = None
        if hasattr(llm_response, 'error') and llm_response.error:
            response_status = "failure"
            error = str(llm_response.error)

        network_logger.log_llm_call(
            source=source,
            model=model,
            latency_seconds=latency,
            response_status=response_status,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error=error,
        )

    return None

class ArithmeticAgent:

    def __init__(self):
        self.system_instruction = load_instructions_file("agents/arithmetic_agent/instructions.txt")
        self.description = load_instructions_file("agents/arithmetic_agent/description.txt")

        self.MCPConnector = MCPConnector(config_file="agents/arithmetic_agent/mcp/mcp_config.json")

        self._user_id = "arithmetic_agent_user"
        self._agent = None
        self._runner = None

    async def initialize(self) -> "ArithmeticAgent":
        """
        异步初始化 agent，构建 LlmAgent 和 Runner 实例。
        需要在事件循环中调用。
        """
        if self._agent is None:
            self._agent = await self._build_agent()
            self._runner = Runner(
                app_name=self._agent.name,
                agent=self._agent,
                artifact_service=InMemoryArtifactService(),
                session_service=InMemorySessionService(),
                memory_service=InMemoryMemoryService(),
            )
        return self

    async def _build_agent(self) -> LlmAgent:
        mcp_tools = await self.MCPConnector.get_tools()

        model = LiteLlm(model="openai/" + os.getenv("OPENAI_MODEL_NAME", "Qwen/Qwen3-8B"))

        return LlmAgent(
            name="arithmetic_agent",
            model=model,
            instruction=self.system_instruction,
            description=self.description,
            before_model_callback=llm_before_model_callback,
            after_model_callback=llm_after_model_callback,
            tools=[
                *mcp_tools
            ]
        )
    
    async def invoke(self, query: str, session_id: str) -> AsyncIterable[dict]:
        """
        Invoke the agent
        Return a stream of updates back to the caller as the agent processes the query

        {
            'is_task_complete': bool,  # Indicates if the task is complete
            'updates': str,  # Updates on the task progress
            'content': str  # Final result of the task if complete
        }
        
        """

        session = await self._runner.session_service.get_session(
            app_name=self._agent.name,
            session_id=session_id,
            user_id=self._user_id,
        )

        if not session:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name,
                session_id=session_id,
                user_id=self._user_id,
            )
        
        user_content = types.Content(
            role="user",
            parts = [types.Part.from_text(text=query)]
        )

        async for event in self._runner.run_async(
            user_id=self._user_id,
            session_id=session_id,
            new_message=user_content
        ):
            print_json_response(event, "================ NEW EVENT ================")
            
            print(f"is_final_response: {event.is_final_response()}")    
            
            if event.is_final_response():
                final_response = ""
                if event.content and event.content.parts and event.content.parts[-1].text:
                    final_response = event.content.parts[-1].text
                
                yield {
                    'is_task_complete': True,
                    'content': final_response
                }
            else:
                yield {
                    'is_task_complete': False,
                    'updates': "Agent is processing your request..."
                }

def print_json_response(response: Any, title: str) -> None:
    # Displays a formatted and color-highlighted view of the response
    print(f"\n=== {title} ===")  # Section title for clarity
    try:
        if hasattr(response, "root"):  # Check if response is wrapped by SDK
            data = response.root.model_dump(mode="json", exclude_none=True)
        else:
            data = response.model_dump(mode="json", exclude_none=True)

        json_str = json.dumps(data, indent=2, ensure_ascii=False)  # Convert dict to pretty JSON string
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)  # Apply syntax highlighting
        rprint(syntax)  # Print it with color
    except Exception as e:
        # Print fallback text if something fails
        rprint(f"[red bold]Error printing JSON:[/red bold] {e}")
        rprint(repr(response))
