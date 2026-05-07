from collections.abc import AsyncIterable
import json
import time
import logging
import os
from typing import Any
from uuid import uuid4
from utilities.a2a.agent_connect import AgentConnector
from utilities.a2a.agent_discovery import AgentDiscovery
from utilities.common.file_loader import load_instructions_file
from google.adk.agents import LlmAgent
from google.adk import Runner
from google.adk.models.lite_llm import LiteLlm

from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.tools.function_tool import FunctionTool

from google.genai import types
from rich import print as rprint
from rich.syntax import Syntax

from utilities.mcp.mcp_connect import MCPConnector
from utilities.common.network_logger import network_logger

from a2a.types import AgentCard

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)


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

class HostAgent:
    """
    Orchestrator agent 
    - Discover A2A agents via agent discovery
    - Discover the MCP servers via MCP connectors and load the MCP tools
    - Routes the user query by picking the correct agent/tool
    """

    def __init__(self):
        self.system_instruction = load_instructions_file("agents/host_agent/instructions.txt")
        self.description = load_instructions_file("agents/host_agent/description.txt")
        
        self.MCPConnector = MCPConnector()
        self.AgentDiscovery = AgentDiscovery()
        
        self._agent = None
        self._user_id = "host_agent_user"
        self._runner = None
        self._tool_call_times: dict[str, tuple[str, float]] = {}  # tool_call_id -> (tool_name, start_time)
        self._tool_call_log_entries: dict[str, Any] = {}  # tool_call_id -> NetworkLogEntry

    async def create(self):
        self._agent = await self._build_agent()
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def _list_agents(self) -> list[dict]:
        """
        A2A tool: returns the list of dictionaries with agent card 
        objects of registered A2A child agents

        Returns:
            list[dict]: List of agent card object dictionaries
        """
        cards = await self.AgentDiscovery.list_agent_cards()

        return [card.model_dump(exclude_none=True) for card in cards]

    async def _delegate_task(self, agent_name: str, message: str) -> str:
        """
        A2A工具，用于通过A2A协议调用其他Agent，把消息发送给其他Agent后，接收其他Agent的执行结果
        参数：
            agent_name: 调用agent的名称
            message: 发送给其他Agent的消息
        返回：
            如果Agent不存在，返回 "Agent not found"；如果Agent存在且执行任务成功，返回执行结果；如果Agent存在但执行失败，返回失败信息
        """

        cards = await self.AgentDiscovery.list_agent_cards()

        matched_card = None
        for card in cards:
            if card.name.lower() == agent_name.lower():
                matched_card = card
            elif getattr(card, "id","").lower() == agent_name.lower():
                matched_card = card

        if matched_card is None:
            return "Agent not found"

        connector = AgentConnector(agent_card=matched_card)

        # 记录 A2A 调用时延和日志
        request_id = str(uuid4())
        log_entry = network_logger.log_a2a_request(
            source="host_agent",
            destination=agent_name,
            request_id=request_id,
            message=message,
        )

        a2a_start = time.time()
        success = True
        result = None
        error_msg = None

        try:
            result = await connector.send_task(message=message, session_id=str(uuid4()))
        except Exception as e:
            success = False
            error_msg = str(e)
            result = f"A2A call failed: {str(e)}"

        a2a_latency = time.time() - a2a_start

        network_logger.log_a2a_response(
            entry=log_entry,
            response_time_seconds=a2a_latency,
            success=success,
            result=result[:500] if result else None,
            error=error_msg
        )

        return result

                
    
    async def _build_agent(self) -> LlmAgent:

        mcp_tools = await self.MCPConnector.get_tools()

        model = LiteLlm(model="openai/" + os.getenv("OPENAI_MODEL_NAME", "Qwen/Qwen3-8B"))

        return LlmAgent(
            name="host_agent",
            model=model,
            instruction=self.system_instruction,
            description=self.description,
            before_model_callback=llm_before_model_callback,
            after_model_callback=llm_after_model_callback,
            tools=[
                FunctionTool(self._delegate_task),
                FunctionTool(self._list_agents),
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
        start_time = time.time()

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

            # MCP 工具调用时延记录
            # 1. function_calls 事件：记录开始时间
            function_calls = event.get_function_calls()
            if function_calls:
                for fc in function_calls:
                    tool_name = getattr(fc, 'name', str(fc))
                    fc_id = getattr(fc, 'id', None)
                    fc_args = getattr(fc, 'args', {})
                    if fc_id:
                        self._tool_call_times[fc_id] = (tool_name, time.time())
                        log_entry = network_logger.log_mcp_request(
                            source="host_agent",
                            destination=tool_name,
                            request_id=fc_id,
                            tool_name=tool_name,
                            arguments=fc_args
                        )
                        self._tool_call_log_entries[fc_id] = log_entry
                    logger.info(f"type: mcp_tool, tool: {tool_name}, action: called")

            # 2. function_responses 事件：计算并记录时延
            function_responses = event.get_function_responses()
            if function_responses:
                for fr in function_responses:
                    tool_name = getattr(fr, 'name', str(fr))
                    fr_id = getattr(fr, 'id', None)
                    latency = None
                    success = True
                    result_data = None

                    if fr_id and fr_id in self._tool_call_times:
                        _, start_time = self._tool_call_times.pop(fr_id)
                        latency = time.time() - start_time

                    log_entry = self._tool_call_log_entries.pop(fr_id, None) if fr_id else None

                    if hasattr(fr, 'response') and fr.response:
                        success = fr.response is not None
                        try:
                            result_data = fr.response.model_dump(mode='json', exclude_none=True)
                        except Exception:
                            result_data = str(fr.response)
                    else:
                        success = False

                    if log_entry and latency is not None:
                        network_logger.log_mcp_response(
                            entry=log_entry,
                            response_time_seconds=latency,
                            success=success,
                            result=result_data,
                            error=None if success else "MCP tool call failed"
                        )

                    logger.info(f"type: mcp_tool, tool: {tool_name}, latency: {latency:.3f}s, success: {success}")

            print(f"is_final_response: {event.is_final_response()}")    
            
            if event.is_final_response():
                end_time = time.time()
                latency = end_time - start_time

                logger.info(msg=f"type: total, query: {query}, latency: {latency}, success: True")

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
