from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater

from agents.arithmetic_agent.agent import ArithmeticAgent
from a2a.utils import (
    new_task,
    new_agent_text_message
)

from a2a.utils.errors import ServerError

from a2a.types import (
    Task,
    TaskState,
    UnsupportedOperationError
)

import asyncio
import time

from utilities.common.network_logger import network_logger

class ArithmeticAgentExecutor(AgentExecutor):
    """
    Implements the AgentExecutor interface to integrate the
    arithmetic agent with the A2A framework.
    """

    def __init__(self):
        self._agent = ArithmeticAgent()

    async def create(self):
        """
        Factory method to create and asynchronously initialize the ArithmeticAgentExecutor.
        """
        await self._agent.initialize()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Executes the agent with the provided context and event queue.
        """
        start_time = time.time()
        query = context.get_user_input()
        task = context.current_task

        # 获取调用方信息
        request_id = ""
        source = "unknown"
        if context.message:
            request_id = getattr(context.message, 'message_id', '') or ''
            # 尝试从 session_id 或其他字段获取调用方信息
            if hasattr(context.message, 'session_id'):
                source = getattr(context.message.session_id, 'user', 'unknown') or 'unknown'

        log_entry = network_logger.log_a2a_handle_request(
            source=source,
            destination="website_builder_simple",
            request_id=request_id,
            message=query,
        )

        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.contextId)

        try:
            async for item in self._agent.invoke(query, task.contextId):
                is_task_complete = item.get("is_task_complete", False)

                if not is_task_complete:
                    message = item.get('updates','The Agent is still working on your request.')
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(message, task.contextId, task.id)
                    )
                else:
                    final_result = item.get('content','no result received')
                    await updater.update_status(
                        TaskState.completed,
                        new_agent_text_message(final_result, task.contextId, task.id)
                    )

                    await asyncio.sleep(0.1)  # Allow time for the message to be processed

                    handling_time = time.time() - start_time
                    network_logger.log_a2a_handle_response(
                        entry=log_entry,
                        handling_time_seconds=handling_time,
                        response_status="success",
                        error=None
                    )

                    break
        except Exception as e:
            handling_time = time.time() - start_time
            error_message = f"An error occurred: {str(e)}"
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(error_message, task.contextId, task.id)
            )
            network_logger.log_a2a_handle_response(
                entry=log_entry,
                handling_time_seconds=handling_time,
                response_status="failure",
                error=str(e)
            )
            raise

    async def cancel(self, request: RequestContext, event_queue: EventQueue) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())
    