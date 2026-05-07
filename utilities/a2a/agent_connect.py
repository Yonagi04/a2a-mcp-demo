from typing import Any
from uuid import uuid4
from a2a.types import (
    AgentCard,
    Task,
    SendMessageRequest,
    MessageSendParams
)
import httpx
from a2a.client import A2AClient
from utilities.common.network_logger import network_logger
import time

class AgentConnector:
    """
    Connects to a remote A2A agent and provides a uniform method to delegate tasks
    """

    def __init__(self, agent_card: AgentCard):
        self.agent_card = agent_card

    async def send_task(self, message: str, session_id: str) -> str:
        """
        Send a task to the agent and return the Task object

        Args:
            message (str): The message to send to the agent
            session_id (str): The session ID for tracking the task

        Returns:
            Task: The Task object containing the response from the agent
        """
        request_id = str(uuid4())
        log_entry = network_logger.log_a2a_request(
            source="host_agent",
            destination=self.agent_card.name,
            request_id=request_id,
            message=message,
            request_data={"session_id": session_id}
        )

        a2a_start = time.time()
        success = True
        agent_response = None
        error_msg = None

        try:
            async with httpx.AsyncClient(timeout=300.0) as httpx_client:
                a2a_client = A2AClient(
                    httpx_client=httpx_client,
                    agent_card=self.agent_card,
                )

                send_message_payload: dict[str, Any] = {
                    'message': {
                        'role': 'user',
                        'messageId': str(uuid4()),
                        'parts': [
                            {
                                'text': message,
                                'kind': 'text'
                            }
                        ]
                    }
                }

                request = SendMessageRequest(
                    id=str(uuid4()),
                    params=MessageSendParams(
                        **send_message_payload
                    )
                )

                response = await a2a_client.send_message(
                    request=request
                )

                response_data = response.model_dump(mode='json', exclude_none=True)

                try:
                    agent_response = response_data['result']['status']['message']['parts'][0]['text']
                except (KeyError, IndexError):
                    agent_response = "No response from agent"
        except Exception as e:
            success = False
            error_msg = str(e)
            agent_response = f"A2A call failed: {str(e)}"

        a2a_latency = time.time() - a2a_start

        network_logger.log_a2a_response(
            entry=log_entry,
            response_time_seconds=a2a_latency,
            success=success,
            result=agent_response[:500] if agent_response else None,
            error=error_msg
        )

        return agent_response
