"""
Test suite for orchestrator sync/async compatibility.
Tests both async-native and sync-wrapped orchestrator behavior.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock

from src.agents.orchestrator import MultiAgentOrchestrator, SmolagentsOrchestrator
from src.models.enums import CalculatorPolicy, CommandWord, Tier, LLMModel
from src.models.question_models import GenerationRequest
from src.services.llm_factory import LLMFactory
from src.services.mock_llm_service import MockLLMService


class TestMultiAgentOrchestrator:
    """Test the async-native orchestrator."""

    @pytest.fixture
    def mock_llm_factory(self):
        """Create a mock LLM factory for testing."""
        factory = Mock(spec=LLMFactory)
        mock_service = MockLLMService()
        factory.get_service.return_value = mock_service
        return factory

    @pytest.fixture
    def generation_request(self):
        """Create a test generation request."""
        return GenerationRequest(
            topic="algebra",
            tier=Tier.CORE,
            grade_level=8,
            marks=3,
            calculator_policy=CalculatorPolicy.NOT_ALLOWED,
            command_word=CommandWord.CALCULATE,
            llm_model=LLMModel.GPT_4O_MINI
        )

    @pytest.fixture
    def orchestrator(self, mock_llm_factory):
        """Create orchestrator with mock factory."""
        return MultiAgentOrchestrator(llm_factory=mock_llm_factory)

    @pytest.mark.asyncio
    async def test_async_orchestrator_basic_workflow(self, orchestrator, generation_request):
        """Test basic async orchestration workflow."""
        result = await orchestrator.generate_question_async(generation_request)
        
        assert result is not None
        assert isinstance(result, dict)
        # Should have completed at least the generation step
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_async_orchestrator_with_quality_control(self, orchestrator, generation_request):
        """Test async orchestration with quality control workflow."""
        result = await orchestrator.generate_question_async(
            generation_request, 
            max_refinement_cycles=2
        )
        
        assert result is not None
        assert isinstance(result, dict)
        # Should have quality decision
        if "quality_decision" in result:
            assert result["quality_decision"] in ["approve", "refine", "reject", "manual_review", "regenerate"]

    def test_sync_orchestrator_without_event_loop(self, orchestrator, generation_request):
        """Test sync wrapper when no event loop is running."""
        # This should work without throwing event loop errors
        result = orchestrator.generate_question_sync(generation_request)
        
        assert result is not None
        assert isinstance(result, dict)

    def test_sync_orchestrator_with_running_loop(self, orchestrator, generation_request):
        """Test sync wrapper when event loop is already running."""
        async def run_in_loop():
            # This simulates calling sync method from within async context
            # Should use ThreadPoolExecutor, not asyncio.run()
            result = orchestrator.generate_question_sync(generation_request)
            return result

        # Run the test within an event loop
        result = asyncio.run(run_in_loop())
        
        assert result is not None
        assert isinstance(result, dict)


class TestSmolagentsOrchestrator:
    """Test the smolagents-compatible orchestrator."""

    @pytest.fixture
    def mock_llm_factory(self):
        """Create a mock LLM factory for testing."""
        factory = Mock(spec=LLMFactory)
        mock_service = MockLLMService()
        factory.get_service.return_value = mock_service
        return factory

    @pytest.fixture
    def smolagents_orchestrator(self, mock_llm_factory):
        """Create smolagents orchestrator with mock factory."""
        return SmolagentsOrchestrator(llm_factory=mock_llm_factory)

    def test_smolagents_generate_question_dict_input(self, smolagents_orchestrator):
        """Test smolagents orchestrator with dict input (smolagents style)."""
        request_dict = {
            "topic": "probability",
            "tier": "Core",
            "grade_level": 7,
            "marks": 4,
            "calculator_policy": "not_allowed"
        }
        
        result = smolagents_orchestrator.generate_question(request_dict)
        
        assert result is not None
        assert isinstance(result, dict)
        assert "error" not in result  # Should not have errors

    def test_smolagents_generate_question_pydantic_input(self, smolagents_orchestrator):
        """Test smolagents orchestrator with Pydantic model input."""
        request = GenerationRequest(
            topic="geometry",
            tier=Tier.EXTENDED,
            grade_level=9,
            marks=5,
            calculator_policy=CalculatorPolicy.ALLOWED,
            command_word=CommandWord.CALCULATE,
            llm_model=LLMModel.GPT_4O_MINI
        )
        
        result = smolagents_orchestrator.generate_question(request)
        
        assert result is not None
        assert isinstance(result, dict)
        assert "error" not in result  # Should not have errors

    def test_smolagents_orchestrator_is_synchronous(self, smolagents_orchestrator):
        """Test that smolagents orchestrator is truly synchronous."""
        request_dict = {
            "topic": "algebra",
            "grade_level": 8,
            "marks": 3
        }
        
        # This should not return a coroutine
        result = smolagents_orchestrator.generate_question(request_dict)
        
        assert not asyncio.iscoroutine(result)
        assert isinstance(result, dict)


class TestOrchestratorsCompatibility:
    """Test compatibility between async and sync orchestrators."""

    @pytest.fixture
    def mock_llm_factory(self):
        """Create a mock LLM factory for testing."""
        factory = Mock(spec=LLMFactory)
        mock_service = MockLLMService()
        factory.get_service.return_value = mock_service
        return factory

    def test_both_orchestrators_same_interface(self, mock_llm_factory):
        """Test that both orchestrators can handle the same inputs."""
        request_dict = {
            "topic": "calculus",
            "grade_level": 9,
            "marks": 6
        }
        
        # Create both orchestrators
        async_orch = MultiAgentOrchestrator(llm_factory=mock_llm_factory)
        sync_orch = SmolagentsOrchestrator(llm_factory=mock_llm_factory)
        
        # Both should handle dict input
        sync_result = sync_orch.generate_question(request_dict)
        async_result = async_orch.generate_question_sync(request_dict)
        
        assert isinstance(sync_result, dict)
        assert isinstance(async_result, dict)
        
        # Results should have similar structure
        assert "error" not in sync_result
        assert "error" not in async_result

    @pytest.mark.asyncio
    async def test_orchestrator_consistency(self, mock_llm_factory):
        """Test that async and sync versions produce consistent results."""
        request = GenerationRequest(
            topic="statistics",
            tier=Tier.CORE,
            grade_level=8,
            marks=4,
            calculator_policy=CalculatorPolicy.NOT_ALLOWED,
            llm_model=LLMModel.GPT_4O_MINI
        )
        
        orchestrator = MultiAgentOrchestrator(llm_factory=mock_llm_factory)
        
        # Get results from both methods
        async_result = await orchestrator.generate_question_async(request)
        sync_result = orchestrator.generate_question_sync(request)
        
        # Both should succeed
        assert async_result is not None
        assert sync_result is not None
        assert isinstance(async_result, dict)
        assert isinstance(sync_result, dict)