"""
Tests for Marker Agent.

Following TDD approach:
1. Write failing tests first (RED)
2. Implement minimal code to pass tests (GREEN)
3. Refactor while keeping tests green (REFACTOR)
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.agents.marker_agent import MarkerAgent, MarkerAgentError
from src.models.enums import CalculatorPolicy
from src.models.question_models import (
    GenerationRequest,
    MarkingCriterion,
    SolutionAndMarkingScheme,
)
from src.services.json_parser import JSONExtractionResult
from src.services.llm_service import LLMError, LLMResponse, LLMTimeoutError


@pytest.fixture
def sample_question():
    """Sample question for marker agent testing"""
    return {
        "question_text": "Calculate the value of 3x + 2 when x = 5",
        "marks": 3,
        "command_word": "Calculate",
        "subject_content_refs": ["C2.1", "C2.2"],
    }


@pytest.fixture
def sample_marking_scheme_json():
    """Sample marking scheme JSON response from LLM"""
    return {
        "total_marks": 3,
        "mark_allocation_criteria": [
            {
                "criterion_text": "Correct substitution of x = 5",
                "marks_value": 1,
                "mark_type": "M",
                "mark_code_display": "M1",
            },
            {
                "criterion_text": "Correct calculation: 3(5) + 2 = 15 + 2",
                "marks_value": 1,
                "mark_type": "M",
                "mark_code_display": "M2",
            },
            {
                "criterion_text": "Final answer: 17",
                "marks_value": 1,
                "mark_type": "A",
                "mark_code_display": "A1",
            },
        ],
        "final_answers": [{"answer_text": "17", "value_numeric": 17.0, "unit": None}],
    }


@pytest.fixture
def sample_generation_config():
    """Sample generation configuration for marker agent"""
    return GenerationRequest(
        topic="algebra",
        marks=3,
        tier="Core",
        grade_level=8,
        calculator_policy=CalculatorPolicy.NOT_ALLOWED,
        subject_content_refs=["C2.1", "C2.2"],
    )


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing"""
    mock = AsyncMock()
    mock.generate = AsyncMock()
    return mock


@pytest.fixture
def prompt_manager():
    """Mock prompt manager"""
    from src.services.prompt_manager import PromptManager

    return PromptManager()


@pytest.fixture
def json_parser():
    """Mock JSON parser"""
    from src.services.json_parser import JSONParser

    return JSONParser()


@pytest.fixture
def agent_config():
    """Sample agent configuration"""
    return {"max_retries": 2, "generation_timeout": 45, "enable_fallback": True}


class TestMarkerAgent:
    """Test suite for MarkerAgent following TDD principles"""

    @pytest.mark.asyncio
    async def test_agent_initialization_with_defaults(self):
        """Test agent initializes with default services when none provided"""
        # GIVEN: No services provided
        # WHEN: Creating agent
        agent = MarkerAgent()

        # THEN: Agent should have default services
        assert agent.name == "Marker"
        assert agent.llm_service is not None
        assert agent.prompt_manager is not None
        assert agent.json_parser is not None
        assert agent.agent_config["max_retries"] == 3
        assert agent.agent_config["generation_timeout"] == 45

    @pytest.mark.asyncio
    async def test_agent_initialization_with_custom_services(
        self, mock_llm_service, prompt_manager, json_parser, agent_config
    ):
        """Test agent accepts custom services via dependency injection"""
        # GIVEN: Custom services
        # WHEN: Creating agent with custom services
        agent = MarkerAgent(
            name="CustomMarker",
            llm_service=mock_llm_service,
            prompt_manager=prompt_manager,
            json_parser=json_parser,
            config=agent_config,
        )

        # THEN: Agent should use provided services
        assert agent.name == "CustomMarker"
        assert agent.llm_service == mock_llm_service
        assert agent.prompt_manager == prompt_manager
        assert agent.json_parser == json_parser
        assert agent.agent_config["max_retries"] == 2  # From custom config

    @pytest.mark.asyncio
    async def test_successful_marking_scheme_generation_end_to_end(
        self,
        mock_llm_service,
        prompt_manager,
        json_parser,
        sample_question,
        sample_marking_scheme_json,
        sample_generation_config,
    ):
        """Test successful marking scheme generation from question to MarkingScheme object"""
        # GIVEN: Agent with mock services
        agent = MarkerAgent(
            llm_service=mock_llm_service, prompt_manager=prompt_manager, json_parser=json_parser
        )

        # AND: Mock LLM returns valid marking scheme JSON
        import json

        mock_llm_service.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(sample_marking_scheme_json),
                model_used="gpt-4o",
                provider="mock",
                tokens_used=200,
                cost_estimate=0.003,
                latency_ms=2500,
            )
        )

        # WHEN: Generating marking scheme
        input_data = {"question": sample_question, "config": sample_generation_config.model_dump()}
        result = await agent.process(input_data)

        # THEN: Should return successful result with marking scheme
        assert result.success is True
        assert result.agent_name == "Marker"
        assert "marking_scheme" in result.output
        assert "generation_metadata" in result.output

        # AND: Marking scheme should have correct properties
        scheme_data = result.output["marking_scheme"]
        assert scheme_data["total_marks_for_part"] == 3
        assert len(scheme_data["mark_allocation_criteria"]) == 3
        assert len(scheme_data["final_answers_summary"]) == 1

    @pytest.mark.asyncio
    async def test_parse_marking_request_valid_input(
        self, sample_question, sample_generation_config
    ):
        """Test parsing valid marking request"""
        # GIVEN: Agent
        agent = MarkerAgent()

        # WHEN: Parsing valid request data
        input_data = {"question": sample_question, "config": sample_generation_config.model_dump()}
        question, config = agent._parse_marking_request(input_data)

        # THEN: Should return valid data
        assert question["question_text"] == sample_question["question_text"]
        assert isinstance(config, GenerationRequest)
        assert config.marks == sample_generation_config.marks

    @pytest.mark.asyncio
    async def test_parse_marking_request_invalid_input(self):
        """Test parsing invalid marking request raises error"""
        # GIVEN: Agent
        agent = MarkerAgent()

        # WHEN: Parsing invalid request (missing required fields)
        invalid_data = {"invalid": "data"}

        # THEN: Should raise MarkerAgentError
        with pytest.raises(MarkerAgentError):
            agent._parse_marking_request(invalid_data)

    @pytest.mark.asyncio
    async def test_generate_cambridge_compliant_marking_criteria(
        self, mock_llm_service, prompt_manager, json_parser, sample_marking_scheme_json
    ):
        """Test that marking criteria follow Cambridge IGCSE standards"""
        # GIVEN: Agent
        agent = MarkerAgent(
            llm_service=mock_llm_service, prompt_manager=prompt_manager, json_parser=json_parser
        )

        # WHEN: Converting JSON to marking scheme
        scheme = agent._convert_to_marking_scheme_object(
            sample_marking_scheme_json, question_marks=3
        )

        # THEN: Should create valid Cambridge-compliant marking scheme
        assert isinstance(scheme, SolutionAndMarkingScheme)
        assert scheme.total_marks_for_part == 3

        # AND: Should have proper mark types
        mark_types = [criterion.mark_type_primary for criterion in scheme.mark_allocation_criteria]
        assert "M" in mark_types  # Method marks
        assert "A" in mark_types  # Accuracy marks

        # AND: Mark codes should follow pattern
        mark_codes = [criterion.mark_code_display for criterion in scheme.mark_allocation_criteria]
        assert any(code.startswith("M") for code in mark_codes)
        assert any(code.startswith("A") for code in mark_codes)

    @pytest.mark.asyncio
    async def test_llm_timeout_triggers_retry(
        self, prompt_manager, json_parser, sample_question, sample_generation_config
    ):
        """Test that LLM timeout triggers retry logic"""
        # GIVEN: Agent with mock LLM that times out then succeeds
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            LLMTimeoutError("Timeout on first attempt"),
            LLMResponse(
                content='{"total_marks": 3, "mark_allocation_criteria": [], "final_answers": []}',
                model_used="gpt-4o",
                provider="mock",
                tokens_used=80,
                cost_estimate=0.001,
                latency_ms=1500,
            ),
        ]

        agent = MarkerAgent(
            llm_service=mock_llm,
            prompt_manager=prompt_manager,
            json_parser=json_parser,
            config={"max_retries": 2},
        )

        # WHEN: Processing request
        input_data = {"question": sample_question, "config": sample_generation_config.model_dump()}
        result = await agent.process(input_data)

        # THEN: Should succeed after retry
        assert result.success is True
        assert mock_llm.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_uses_fallback(
        self, prompt_manager, json_parser, sample_question, sample_generation_config
    ):
        """Test fallback method when max retries exceeded"""
        # GIVEN: Agent with LLM that always fails primary method
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            LLMTimeoutError("Primary attempt 1 failed"),
            LLMTimeoutError("Primary attempt 2 failed"),
            # Fallback succeeds
            LLMResponse(
                content='{"total_marks": 3, "mark_allocation_criteria": [], "final_answers": []}',
                model_used="gpt-4o-mini",
                provider="mock",
                tokens_used=50,
                cost_estimate=0.0005,
                latency_ms=800,
            ),
        ]

        agent = MarkerAgent(
            llm_service=mock_llm,
            prompt_manager=prompt_manager,
            json_parser=json_parser,
            config={"max_retries": 2, "enable_fallback": True},
        )

        # WHEN: Processing request
        input_data = {"question": sample_question, "config": sample_generation_config.model_dump()}
        result = await agent.process(input_data)

        # THEN: Should succeed using fallback
        assert result.success is True
        assert mock_llm.generate.call_count == 3  # 2 primary + 1 fallback

    @pytest.mark.asyncio
    async def test_json_extraction_failure_triggers_retry(
        self, mock_llm_service, prompt_manager, sample_question, sample_generation_config
    ):
        """Test that JSON extraction failure triggers retry"""
        # GIVEN: Agent with JSON parser that fails then succeeds
        mock_json_parser = AsyncMock()
        mock_json_parser.extract_json.side_effect = [
            JSONExtractionResult(success=False, error="Invalid JSON structure"),
            JSONExtractionResult(
                success=True,
                data={"total_marks": 3, "mark_allocation_criteria": [], "final_answers": []},
                extraction_method="test",
            ),
        ]

        agent = MarkerAgent(
            llm_service=mock_llm_service,
            prompt_manager=prompt_manager,
            json_parser=mock_json_parser,
            config={"max_retries": 2},
        )

        # WHEN: Processing request
        input_data = {"question": sample_question, "config": sample_generation_config.model_dump()}
        result = await agent.process(input_data)

        # THEN: Should succeed after retry
        assert result.success is True
        assert mock_json_parser.extract_json.call_count == 2

    @pytest.mark.asyncio
    async def test_marking_scheme_validation_failure(
        self, mock_llm_service, prompt_manager, json_parser
    ):
        """Test marking scheme validation catches invalid schemes"""
        # GIVEN: Agent and request that produces invalid marking scheme
        agent = MarkerAgent(
            llm_service=mock_llm_service, prompt_manager=prompt_manager, json_parser=json_parser
        )

        # AND: Mock LLM returns invalid marking scheme data
        mock_llm_service.generate = AsyncMock(
            return_value=LLMResponse(
                content='{"total_marks": 0, "mark_allocation_criteria": []}',  # Invalid: 0 marks
                model_used="gpt-4o",
                provider="mock",
                tokens_used=50,
                cost_estimate=0.001,
                latency_ms=1000,
            )
        )

        # WHEN: Processing request
        input_data = {
            "question": {"question_text": "Test question", "marks": 3},
            "config": {"marks": 3, "tier": "Core", "topic": "algebra"},
        }
        result = await agent.process(input_data)

        # THEN: Should fail with validation error
        assert result.success is False
        assert "failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_mark_type_assignment_logic(self, sample_marking_scheme_json):
        """Test that mark types are assigned correctly according to Cambridge standards"""
        # GIVEN: Agent
        agent = MarkerAgent()

        # WHEN: Converting marking scheme with various criteria
        scheme = agent._convert_to_marking_scheme_object(
            sample_marking_scheme_json, question_marks=3
        )

        # THEN: Should assign mark types correctly
        criteria = scheme.mark_allocation_criteria

        # Method marks for process/substitution
        method_marks = [c for c in criteria if c.mark_type_primary == "M"]
        assert len(method_marks) >= 1

        # Accuracy marks for final answer
        accuracy_marks = [c for c in criteria if c.mark_type_primary == "A"]
        assert len(accuracy_marks) >= 1

    @pytest.mark.asyncio
    async def test_marking_scheme_object_conversion_from_json(self, sample_marking_scheme_json):
        """Test converting JSON response to SolutionAndMarkingScheme object"""
        # GIVEN: Agent and sample JSON data
        agent = MarkerAgent()

        # WHEN: Converting JSON to marking scheme object
        scheme = agent._convert_to_marking_scheme_object(
            sample_marking_scheme_json, question_marks=3
        )

        # THEN: Should create valid SolutionAndMarkingScheme object
        assert isinstance(scheme, SolutionAndMarkingScheme)
        assert scheme.total_marks_for_part == 3
        assert len(scheme.mark_allocation_criteria) == 3
        assert len(scheme.final_answers_summary) == 1

        # AND: Criteria should have correct structure
        first_criterion = scheme.mark_allocation_criteria[0]
        assert isinstance(first_criterion, MarkingCriterion)
        assert first_criterion.criterion_text is not None
        assert first_criterion.marks_value > 0
        assert first_criterion.mark_type_primary in ["M", "A", "B", "FT"]

    @pytest.mark.asyncio
    async def test_marking_scheme_object_conversion_fallback_mode(self):
        """Test MarkingScheme object creation in fallback mode with minimal data"""
        # GIVEN: Agent and minimal JSON data
        agent = MarkerAgent()
        minimal_json = {"total_marks": 2, "mark_allocation_criteria": [], "final_answers": []}

        # WHEN: Converting with fallback flag
        scheme = agent._convert_to_marking_scheme_object(
            minimal_json, question_marks=2, is_fallback=True
        )

        # THEN: Should create valid MarkingScheme object with defaults
        assert isinstance(scheme, SolutionAndMarkingScheme)
        assert scheme.total_marks_for_part == 2

        # Should create default criteria if none provided
        assert len(scheme.mark_allocation_criteria) >= 1
        assert len(scheme.final_answers_summary) >= 1

    @pytest.mark.asyncio
    async def test_agent_reasoning_steps_logged(
        self,
        prompt_manager,
        json_parser,
        sample_question,
        sample_generation_config,
        sample_marking_scheme_json,
    ):
        """Test that agent logs reasoning steps during execution"""
        # GIVEN: Agent with properly mocked LLM service
        mock_llm = AsyncMock()
        import json

        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(sample_marking_scheme_json),
                model_used="gpt-4o",
                provider="mock",
                tokens_used=150,
                cost_estimate=0.002,
                latency_ms=2000,
            )
        )

        agent = MarkerAgent(
            llm_service=mock_llm, prompt_manager=prompt_manager, json_parser=json_parser
        )

        # WHEN: Processing request
        input_data = {"question": sample_question, "config": sample_generation_config.model_dump()}
        result = await agent.process(input_data)

        # THEN: Should have logged reasoning steps
        assert len(result.reasoning_steps) > 0
        reasoning_text = " ".join(result.reasoning_steps)
        assert "parsing marking request" in reasoning_text.lower()
        assert "generation attempt" in reasoning_text.lower()
        assert "successfully generated" in reasoning_text.lower()

    @pytest.mark.asyncio
    async def test_marking_stats_tracking(self):
        """Test that agent tracks marking generation statistics"""
        # GIVEN: Agent
        agent = MarkerAgent()

        # WHEN: Getting marking stats
        stats = agent.get_marking_stats()

        # THEN: Should return stats dictionary
        assert isinstance(stats, dict)
        assert "total_markings" in stats
        assert "success_rate" in stats
        assert "average_marking_time" in stats

    @pytest.mark.asyncio
    async def test_error_handling_preserves_reasoning_steps(
        self, prompt_manager, json_parser, sample_question, sample_generation_config
    ):
        """Test that errors preserve reasoning steps for debugging"""
        # GIVEN: Agent with failing LLM
        failing_llm = AsyncMock()
        failing_llm.generate.side_effect = LLMError("LLM service unavailable")

        agent = MarkerAgent(
            llm_service=failing_llm,
            prompt_manager=prompt_manager,
            json_parser=json_parser,
            config={"max_retries": 1, "enable_fallback": False},
        )

        # WHEN: Processing request that will fail
        input_data = {"question": sample_question, "config": sample_generation_config.model_dump()}
        result = await agent.process(input_data)

        # THEN: Should preserve reasoning steps even on failure
        assert result.success is False
        assert len(result.reasoning_steps) > 0
        assert any("parsing marking request" in step.lower() for step in result.reasoning_steps)
        assert "failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_agent_config_override(self):
        """Test that agent configuration can be overridden"""
        # GIVEN: Custom configuration
        custom_config = {
            "max_retries": 5,
            "generation_timeout": 90,
            "quality_threshold": 0.8,
            "enable_fallback": False,
        }

        # WHEN: Creating agent with custom config
        agent = MarkerAgent(config=custom_config)

        # THEN: Should use custom configuration
        assert agent.agent_config["max_retries"] == 5
        assert agent.agent_config["generation_timeout"] == 90
        assert agent.agent_config["quality_threshold"] == 0.8
        assert agent.agent_config["enable_fallback"] is False


class TestMarkerAgentIntegration:
    """Integration tests for Marker Agent with real services"""

    @pytest.mark.asyncio
    async def test_integration_with_mock_services(self, prompt_manager, json_parser):
        """Test integration between all services working together"""
        # GIVEN: Agent with properly configured mock LLM service
        mock_llm = AsyncMock()
        # Configure mock to return valid marking scheme JSON
        marking_scheme_json = {
            "total_marks": 4,
            "mark_allocation_criteria": [
                {"criterion_text": "Correct formula A = πr²", "marks_value": 1, "mark_type": "M"},
                {
                    "criterion_text": "Correct substitution r = 5",
                    "marks_value": 1,
                    "mark_type": "M",
                },
                {"criterion_text": "Correct calculation", "marks_value": 1, "mark_type": "M"},
                {"criterion_text": "Final answer with units", "marks_value": 1, "mark_type": "A"},
            ],
            "final_answers": [{"answer_text": "78.5 cm²", "value_numeric": 78.5, "unit": "cm²"}],
        }

        import json

        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(marking_scheme_json),
                model_used="gpt-4o",
                provider="mock",
                tokens_used=200,
                cost_estimate=0.003,
                latency_ms=2500,
            )
        )

        agent = MarkerAgent(
            llm_service=mock_llm, prompt_manager=prompt_manager, json_parser=json_parser
        )

        # WHEN: Processing a complete request
        request_data = {
            "question": {
                "question_text": "Find the area of a circle with radius 5 cm",
                "marks": 4,
                "command_word": "Find",
                "subject_content_refs": ["C5.3"],
            },
            "config": {
                "topic": "mensuration",
                "tier": "Core",
                "grade_level": 8,
                "marks": 4,
                "calculator_policy": "allowed",
            },
        }

        result = await agent.process(request_data)

        # THEN: Should produce valid result
        assert result.success is True
        assert "marking_scheme" in result.output

        # AND: Marking scheme should be properly structured
        scheme = result.output["marking_scheme"]
        assert scheme["total_marks_for_part"] == 4
        assert len(scheme["mark_allocation_criteria"]) > 0


class TestMarkerAgentPerformance:
    """Performance and reliability tests"""

    @pytest.mark.asyncio
    async def test_concurrent_marking_generation_handling(self, prompt_manager, json_parser):
        """Test agent handles concurrent marking generation requests"""
        # GIVEN: Agent with properly configured mock LLM service
        mock_llm = AsyncMock()
        # Configure mock to return valid marking scheme JSON for 2-mark question
        marking_scheme_json = {
            "total_marks": 2,
            "mark_allocation_criteria": [
                {"criterion_text": "Correct method", "marks_value": 1, "mark_type": "M"},
                {"criterion_text": "Correct answer", "marks_value": 1, "mark_type": "A"},
            ],
            "final_answers": [{"answer_text": "36", "value_numeric": 36.0, "unit": None}],
        }

        import json

        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(marking_scheme_json),
                model_used="gpt-4o",
                provider="mock",
                tokens_used=120,
                cost_estimate=0.002,
                latency_ms=1500,
            )
        )

        agent = MarkerAgent(
            llm_service=mock_llm, prompt_manager=prompt_manager, json_parser=json_parser
        )

        request_data = {
            "question": {
                "question_text": "Calculate 15% of 240",
                "marks": 2,
                "command_word": "Calculate",
            },
            "config": {"topic": "percentages", "marks": 2, "tier": "Core"},
        }

        # WHEN: Processing multiple requests concurrently
        tasks = [agent.process(request_data) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # THEN: All requests should succeed
        assert len(results) == 5
        assert all(result.success for result in results)

    @pytest.mark.asyncio
    async def test_marking_timeout_handling(
        self, prompt_manager, json_parser, sample_question, sample_generation_config
    ):
        """Test agent respects marking generation timeout"""
        # GIVEN: Agent with very short timeout
        slow_llm = AsyncMock()

        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(2)  # Slower than timeout

        slow_llm.generate.side_effect = slow_generate

        agent = MarkerAgent(
            llm_service=slow_llm,
            prompt_manager=prompt_manager,
            json_parser=json_parser,
            config={"generation_timeout": 1, "max_retries": 1, "enable_fallback": False},
        )

        # WHEN: Processing request
        input_data = {"question": sample_question, "config": sample_generation_config.model_dump()}
        result = await agent.process(input_data)

        # THEN: Should fail due to timeout
        assert result.success is False
        assert "timeout" in result.error.lower() or "failed" in result.error.lower()
