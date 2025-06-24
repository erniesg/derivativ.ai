"""
Base agent class for Derivativ AI multi-agent system.
Implements smolagents integration patterns with Modal deployment support.
"""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel

from src.models.question_models import AgentResult


class AgentObservation(BaseModel):
    """Single observation made by an agent"""

    type: str  # "observation", "thought", "action"
    content: str
    timestamp: float
    data: Optional[Any] = None


class BaseAgent(ABC):
    """
    Base class for all Derivativ AI agents.
    Implements smolagents patterns with reasoning transparency.
    """

    def __init__(self, name: str, llm_interface, config: dict = None):
        self.name = name
        self.llm = llm_interface
        self.config = config or {}
        self.reasoning_steps: list[AgentObservation] = []
        self.logger = logging.getLogger(f"agent.{name}")
        self.start_time = None

    async def process(self, input_data: dict) -> AgentResult:
        """
        Main processing method for agent operations.
        Implements the observe-think-act pattern with error handling.
        """
        self.start_time = time.time()
        self.reasoning_steps.clear()

        try:
            self._observe(f"Processing request: {input_data}")
            self._think("Analyzing requirements and planning approach")

            output = await self._execute(input_data)

            self._act("Generated output successfully", output)

            return AgentResult(
                success=True,
                agent_name=self.name,
                output=output,
                reasoning_steps=[step.content for step in self.reasoning_steps],
                processing_time=self._get_timing(),
                metadata={"config": self.config},
            )

        except Exception as e:
            return self._handle_error(e)

    @abstractmethod
    async def _execute(self, input_data: dict) -> dict:
        """
        Implement agent-specific logic here.
        Must be overridden by subclasses.
        """
        pass

    def _observe(self, observation: str, data: Any = None):
        """Log an observation made by the agent"""
        self.reasoning_steps.append(
            AgentObservation(
                type="observation", content=observation, timestamp=time.time(), data=data
            )
        )
        self.logger.info(f"[OBSERVE] {observation}")

    def _think(self, reasoning: str):
        """Log agent reasoning/thinking process"""
        self.reasoning_steps.append(
            AgentObservation(type="thought", content=reasoning, timestamp=time.time())
        )
        self.logger.info(f"[THINK] {reasoning}")

    def _act(self, action: str, result: Any = None):
        """Log an action taken by the agent"""
        self.reasoning_steps.append(
            AgentObservation(type="action", content=action, timestamp=time.time(), data=result)
        )
        self.logger.info(f"[ACT] {action}")

    def _handle_error(self, error: Exception) -> AgentResult:
        """Handle errors with proper logging and fallback"""
        error_msg = f"Agent {self.name} failed: {error!s}"
        self.logger.error(error_msg, exc_info=True)

        return AgentResult(
            success=False,
            agent_name=self.name,
            error=error_msg,
            reasoning_steps=[step.content for step in self.reasoning_steps],
            processing_time=self._get_timing(),
            metadata={"error_type": type(error).__name__},
        )

    def _get_timing(self) -> float:
        """Get processing time since start"""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0

    def get_reasoning_summary(self) -> str:
        """Get a formatted summary of agent reasoning"""
        summary = f"=== {self.name} Reasoning ===\n"
        for step in self.reasoning_steps:
            timestamp = datetime.fromtimestamp(step.timestamp).strftime("%H:%M:%S")
            summary += f"[{timestamp}] {step.type.upper()}: {step.content}\n"
        return summary
