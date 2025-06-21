"""
Synchronous wrapper for async agents to support both sync/async usage.
Particularly useful for smolagents integration which is synchronous.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from .base_agent import BaseAgent, AgentResult


class SyncAgentWrapper:
    """
    Wraps async agents to provide synchronous interface.
    Useful for integrating with smolagents or other sync frameworks.
    """
    
    def __init__(self, agent: BaseAgent):
        """
        Initialize wrapper with an async agent.
        
        Args:
            agent: The async agent to wrap
        """
        self.agent = agent
        self._loop = None
        self._thread_executor = ThreadPoolExecutor(max_workers=1)
    
    def process(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Synchronous version of agent.process().
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            AgentResult from the agent
        """
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, need to run in thread
            future = self._thread_executor.submit(self._run_in_new_loop, input_data)
            return future.result()
        except RuntimeError:
            # No event loop, we can create one
            return asyncio.run(self.agent.process(input_data))
    
    def _run_in_new_loop(self, input_data: Dict[str, Any]) -> AgentResult:
        """Run agent in a new event loop in a separate thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.agent.process(input_data))
        finally:
            loop.close()
    
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the wrapped agent."""
        return getattr(self.agent, name)


def make_sync_agent(agent_class: type[BaseAgent], *args, **kwargs) -> SyncAgentWrapper:
    """
    Factory function to create a synchronous version of an agent.
    
    Args:
        agent_class: The agent class to instantiate
        *args, **kwargs: Arguments for agent initialization
        
    Returns:
        SyncAgentWrapper instance
    """
    agent = agent_class(*args, **kwargs)
    return SyncAgentWrapper(agent)


# Convenience functions for each agent type
def create_sync_question_generator(**kwargs) -> SyncAgentWrapper:
    """Create synchronous QuestionGeneratorAgent."""
    from .question_generator import QuestionGeneratorAgent
    return make_sync_agent(QuestionGeneratorAgent, **kwargs)


def create_sync_marker(**kwargs) -> SyncAgentWrapper:
    """Create synchronous MarkerAgent."""
    from .marker_agent import MarkerAgent
    return make_sync_agent(MarkerAgent, **kwargs)


def create_sync_reviewer(**kwargs) -> SyncAgentWrapper:
    """Create synchronous ReviewAgent."""
    from .review_agent import ReviewAgent
    return make_sync_agent(ReviewAgent, **kwargs)


def create_sync_refiner(**kwargs) -> SyncAgentWrapper:
    """Create synchronous RefinementAgent."""
    from .refinement_agent import RefinementAgent
    return make_sync_agent(RefinementAgent, **kwargs)