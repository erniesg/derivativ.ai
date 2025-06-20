"""
Tests for Question Generator Agent.

Following TDD approach:
1. Write failing tests first
2. Implement minimal code to pass tests
3. Refactor while keeping tests green
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.agents.question_generator import QuestionGeneratorAgent, QuestionGeneratorError
from src.models.enums import CommandWord
from src.models.question_models import GenerationRequest, Question
from src.services.json_parser import JSONExtractionResult
from src.services.llm_service import LLMError, LLMResponse, LLMTimeoutError


class TestQuestionGeneratorAgent:
    """Test suite for QuestionGeneratorAgent following TDD principles"""

    @pytest.mark.asyncio
    async def test_agent_initialization_with_defaults(self):
        """Test agent initializes with default services when none provided"""
        # GIVEN: No services provided
        # WHEN: Creating agent
        agent = QuestionGeneratorAgent()

        # THEN: Agent should have default services
        assert agent.name == "QuestionGenerator"
        assert agent.llm_service is not None
        assert agent.prompt_manager is not None
        assert agent.json_parser is not None
        assert agent.agent_config["max_retries"] == 3
        assert agent.agent_config["generation_timeout"] == 60

    @pytest.mark.asyncio
    async def test_agent_initialization_with_custom_services(
        self, mock_llm_service, prompt_manager, json_parser, agent_config
    ):
        """Test agent accepts custom services via dependency injection"""
        # GIVEN: Custom services
        # WHEN: Creating agent with custom services
        agent = QuestionGeneratorAgent(
            name="CustomGenerator",
            llm_service=mock_llm_service,
            prompt_manager=prompt_manager,
            json_parser=json_parser,
            config=agent_config,
        )

        # THEN: Agent should use provided services
        assert agent.name == "CustomGenerator"
        assert agent.llm_service == mock_llm_service
        assert agent.prompt_manager == prompt_manager
        assert agent.json_parser == json_parser
        assert agent.agent_config["max_retries"] == 2  # From custom config

    @pytest.mark.asyncio
    async def test_successful_question_generation_end_to_end(
        self,
        mock_llm_service,
        prompt_manager,
        json_parser,
        sample_generation_request,
        sample_question_json,
    ):
        """Test successful question generation from request to Question object"""
        # GIVEN: Agent with mock services
        agent = QuestionGeneratorAgent(
            llm_service=mock_llm_service, prompt_manager=prompt_manager, json_parser=json_parser
        )

        # AND: Mock LLM returns valid JSON
        import json

        mock_llm_service.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(sample_question_json),
                model="gpt-4o",
                provider="mock",
                tokens_used=150,
                cost_estimate=0.002,
                generation_time=2.0,
            )
        )

        # WHEN: Processing generation request
        input_data = sample_generation_request.model_dump()
        result = await agent.process(input_data)

        # THEN: Should return successful result with question
        assert result.success is True
        assert result.agent_name == "QuestionGenerator"
        assert "question" in result.output
        assert "generation_metadata" in result.output

        # AND: Question should have correct properties
        question_data = result.output["question"]
        assert question_data["marks"] == sample_generation_request.marks
        assert question_data["command_word"] == sample_generation_request.command_word.value
        assert question_data["raw_text_content"] == sample_question_json["question_text"]

    @pytest.mark.asyncio
    async def test_parse_generation_request_valid_input(self, sample_generation_request):
        """Test parsing valid generation request"""
        # GIVEN: Agent
        agent = QuestionGeneratorAgent()

        # WHEN: Parsing valid request data
        input_data = sample_generation_request.model_dump()
        request = agent._parse_generation_request(input_data)

        # THEN: Should return valid GenerationRequest
        assert isinstance(request, GenerationRequest)
        assert request.topic == sample_generation_request.topic
        assert request.marks == sample_generation_request.marks
        assert request.command_word == sample_generation_request.command_word

    @pytest.mark.asyncio
    async def test_parse_generation_request_invalid_input(self):
        """Test parsing invalid generation request raises error"""
        # GIVEN: Agent
        agent = QuestionGeneratorAgent()

        # WHEN: Parsing invalid request (missing required fields)
        invalid_data = {"invalid": "data"}

        # THEN: Should raise QuestionGeneratorError
        with pytest.raises(QuestionGeneratorError):
            agent._parse_generation_request(invalid_data)

    @pytest.mark.asyncio
    async def test_parse_generation_request_invalid_marks(self):
        """Test parsing request with invalid mark values"""
        # GIVEN: Agent
        agent = QuestionGeneratorAgent()

        # WHEN: Parsing request with invalid marks
        invalid_data = {
            "topic": "algebra",
            "marks": 25,  # Invalid: too high
            "tier": "Core",  # Valid tier enum value
        }

        # THEN: Should raise QuestionGeneratorError
        with pytest.raises(QuestionGeneratorError, match="Marks must be between 1 and 20"):
            agent._parse_generation_request(invalid_data)

    @pytest.mark.asyncio
    async def test_llm_timeout_triggers_retry(
        self, prompt_manager, json_parser, sample_generation_request
    ):
        """Test that LLM timeout triggers retry logic"""
        # GIVEN: Agent with mock LLM that times out then succeeds
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            LLMTimeoutError("Timeout on first attempt"),
            LLMResponse(
                content='{"question_text": "Test question", "marks": 2}',
                model="gpt-4o",
                provider="mock",
                tokens_used=50,
                cost_estimate=0.001,
                generation_time=1.0,
            ),
        ]

        agent = QuestionGeneratorAgent(
            llm_service=mock_llm,
            prompt_manager=prompt_manager,
            json_parser=json_parser,
            config={"max_retries": 2},
        )

        # WHEN: Processing request
        input_data = sample_generation_request.model_dump()
        result = await agent.process(input_data)

        # THEN: Should succeed after retry
        assert result.success is True
        assert mock_llm.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_uses_fallback(
        self, prompt_manager, json_parser, sample_generation_request
    ):
        """Test fallback method when max retries exceeded"""
        # GIVEN: Agent with LLM that always fails primary method
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            LLMTimeoutError("Primary attempt 1 failed"),
            LLMTimeoutError("Primary attempt 2 failed"),
            # Fallback succeeds
            LLMResponse(
                content='{"question_text": "Fallback question", "marks": 2}',
                model="gpt-4o-mini",
                provider="mock",
                tokens_used=30,
                cost_estimate=0.0005,
                generation_time=0.5,
            ),
        ]

        agent = QuestionGeneratorAgent(
            llm_service=mock_llm,
            prompt_manager=prompt_manager,
            json_parser=json_parser,
            config={"max_retries": 2, "enable_fallback": True},
        )

        # WHEN: Processing request
        input_data = sample_generation_request.model_dump()
        result = await agent.process(input_data)

        # THEN: Should succeed using fallback
        assert result.success is True
        assert mock_llm.generate.call_count == 3  # 2 primary + 1 fallback
        assert "Fallback question" in str(result.output)

    @pytest.mark.asyncio
    async def test_json_extraction_failure_triggers_retry(
        self, mock_llm_service, prompt_manager, sample_generation_request
    ):
        """Test that JSON extraction failure triggers retry"""
        # GIVEN: Agent with JSON parser that fails then succeeds
        mock_json_parser = AsyncMock()
        mock_json_parser.extract_json.side_effect = [
            JSONExtractionResult(success=False, error="Invalid JSON"),
            JSONExtractionResult(
                success=True,
                data={"question_text": "Valid question", "marks": 2},
                extraction_method="test",
            ),
        ]

        agent = QuestionGeneratorAgent(
            llm_service=mock_llm_service,
            prompt_manager=prompt_manager,
            json_parser=mock_json_parser,
            config={"max_retries": 2},
        )

        # WHEN: Processing request
        input_data = sample_generation_request.model_dump()
        result = await agent.process(input_data)

        # THEN: Should succeed after retry
        assert result.success is True
        assert mock_json_parser.extract_json.call_count == 2

    @pytest.mark.asyncio
    async def test_question_validation_failure(self, mock_llm_service, prompt_manager, json_parser):
        """Test question validation catches invalid questions"""
        # GIVEN: Agent and request that produces invalid question
        agent = QuestionGeneratorAgent(
            llm_service=mock_llm_service, prompt_manager=prompt_manager, json_parser=json_parser
        )

        # AND: Mock LLM returns invalid question data
        mock_llm_service.generate = AsyncMock(
            return_value=LLMResponse(
                content='{"question_text": "", "marks": 0}',  # Invalid: empty text, 0 marks
                model="gpt-4o",
                provider="mock",
                tokens_used=50,
                cost_estimate=0.001,
                generation_time=1.0,
            )
        )

        # WHEN: Processing request
        input_data = {"topic": "algebra", "marks": 3, "tier": "Core"}
        result = await agent.process(input_data)

        # THEN: Should fail with validation error
        assert result.success is False
        assert (
            "failed" in result.error.lower()
        )  # Question validation failure leads to retry exhaustion

    @pytest.mark.asyncio
    async def test_multiple_question_generation(
        self, mock_llm_service, prompt_manager, json_parser, sample_question_json
    ):
        """Test generating multiple questions concurrently"""
        # GIVEN: Agent
        agent = QuestionGeneratorAgent(
            llm_service=mock_llm_service, prompt_manager=prompt_manager, json_parser=json_parser
        )

        # AND: Request for multiple questions
        request = GenerationRequest(
            topic="algebra",
            marks=2,
            count=3,  # Request 3 questions
            tier="Core",
        )

        # WHEN: Generating multiple questions
        questions = await agent.generate_multiple_questions(request)

        # THEN: Should return requested number of questions
        assert len(questions) == 3
        assert all(isinstance(q, Question) for q in questions)
        assert all(q.marks == 2 for q in questions)

    @pytest.mark.asyncio
    async def test_question_object_conversion_from_json(
        self, sample_question_json, sample_generation_request
    ):
        """Test converting JSON response to Question object"""
        # GIVEN: Agent and sample JSON data
        agent = QuestionGeneratorAgent()

        # WHEN: Converting JSON to Question object
        question = await agent._convert_to_question_object(
            sample_question_json, sample_generation_request
        )

        # THEN: Should create valid Question object
        assert isinstance(question, Question)
        assert question.raw_text_content == sample_question_json["question_text"]
        assert question.marks == sample_question_json["marks"]
        assert question.command_word.value == sample_question_json["command_word"]
        assert len(question.solution_and_marking_scheme.mark_allocation_criteria) > 0
        assert len(question.solver_algorithm.steps) > 0

    @pytest.mark.asyncio
    async def test_question_object_conversion_fallback_mode(self, sample_generation_request):
        """Test Question object creation in fallback mode with minimal data"""
        # GIVEN: Agent and minimal JSON data
        agent = QuestionGeneratorAgent()
        minimal_json = {"question_text": "Simple question", "marks": 2}

        # WHEN: Converting with fallback flag
        question = await agent._convert_to_question_object(
            minimal_json, sample_generation_request, is_fallback=True
        )

        # THEN: Should create valid Question object with defaults
        assert isinstance(question, Question)
        assert question.raw_text_content == "Simple question"
        assert question.marks == 2
        assert question.command_word == CommandWord.CALCULATE  # Default
        assert (
            len(question.solution_and_marking_scheme.mark_allocation_criteria) == 1
        )  # Default criterion

    @pytest.mark.asyncio
    async def test_agent_reasoning_steps_logged(
        self, mock_llm_service, prompt_manager, json_parser, sample_generation_request
    ):
        """Test that agent logs reasoning steps during execution"""
        # GIVEN: Agent
        agent = QuestionGeneratorAgent(
            llm_service=mock_llm_service, prompt_manager=prompt_manager, json_parser=json_parser
        )

        # WHEN: Processing request
        input_data = sample_generation_request.model_dump()
        result = await agent.process(input_data)

        # THEN: Should have logged reasoning steps
        assert len(result.reasoning_steps) > 0
        reasoning_text = " ".join(result.reasoning_steps)
        assert "parsing generation request" in reasoning_text.lower()
        assert "generation attempt" in reasoning_text.lower()
        assert "successfully generated" in reasoning_text.lower()

    @pytest.mark.asyncio
    async def test_generation_stats_tracking(self):
        """Test that agent tracks generation statistics"""
        # GIVEN: Agent
        agent = QuestionGeneratorAgent()

        # WHEN: Getting generation stats
        stats = agent.get_generation_stats()

        # THEN: Should return stats dictionary
        assert isinstance(stats, dict)
        assert "total_generations" in stats
        assert "success_rate" in stats
        assert "average_generation_time" in stats

    @pytest.mark.asyncio
    async def test_error_handling_preserves_reasoning_steps(
        self, prompt_manager, json_parser, sample_generation_request
    ):
        """Test that errors preserve reasoning steps for debugging"""
        # GIVEN: Agent with failing LLM
        failing_llm = AsyncMock()
        failing_llm.generate.side_effect = LLMError("LLM service unavailable")

        agent = QuestionGeneratorAgent(
            llm_service=failing_llm,
            prompt_manager=prompt_manager,
            json_parser=json_parser,
            config={"max_retries": 1, "enable_fallback": False},
        )

        # WHEN: Processing request that will fail
        input_data = sample_generation_request.model_dump()
        result = await agent.process(input_data)

        # THEN: Should preserve reasoning steps even on failure
        assert result.success is False
        assert len(result.reasoning_steps) > 0
        assert any("parsing generation request" in step.lower() for step in result.reasoning_steps)
        assert "failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_agent_config_override(self):
        """Test that agent configuration can be overridden"""
        # GIVEN: Custom configuration
        custom_config = {
            "max_retries": 5,
            "generation_timeout": 120,
            "quality_threshold": 0.9,
            "enable_fallback": False,
        }

        # WHEN: Creating agent with custom config
        agent = QuestionGeneratorAgent(config=custom_config)

        # THEN: Should use custom configuration
        assert agent.agent_config["max_retries"] == 5
        assert agent.agent_config["generation_timeout"] == 120
        assert agent.agent_config["quality_threshold"] == 0.9
        assert agent.agent_config["enable_fallback"] is False


class TestQuestionGeneratorIntegration:
    """Integration tests for QuestionGenerator with real services"""

    @pytest.mark.asyncio
    async def test_integration_with_mock_services(
        self, mock_llm_service, prompt_manager, json_parser
    ):
        """Test integration between all services working together"""
        # GIVEN: Agent with real service instances
        agent = QuestionGeneratorAgent(
            llm_service=mock_llm_service, prompt_manager=prompt_manager, json_parser=json_parser
        )

        # WHEN: Processing a complete request
        request_data = {
            "topic": "quadratic equations",
            "tier": "Extended",
            "grade_level": 9,
            "marks": 4,
            "calculator_policy": "allowed",
            "subject_content_refs": ["C2.4"],
            "command_word": "Find",
        }

        result = await agent.process(request_data)

        # THEN: Should produce valid result
        assert result.success is True
        assert "question" in result.output

        # AND: Question should be properly structured
        question = result.output["question"]
        assert question["marks"] == 4
        assert question["command_word"] == "Find"
        assert len(question["raw_text_content"]) > 10


class TestQuestionGeneratorPerformance:
    """Performance and reliability tests"""

    @pytest.mark.asyncio
    async def test_concurrent_generation_handling(
        self, mock_llm_service, prompt_manager, json_parser
    ):
        """Test agent handles concurrent generation requests"""
        # GIVEN: Agent and multiple requests
        agent = QuestionGeneratorAgent(
            llm_service=mock_llm_service, prompt_manager=prompt_manager, json_parser=json_parser
        )

        request_data = {"topic": "algebra", "marks": 2, "tier": "Core"}

        # WHEN: Processing multiple requests concurrently
        tasks = [agent.process(request_data) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # THEN: All requests should succeed
        assert len(results) == 5
        assert all(result.success for result in results)

    @pytest.mark.asyncio
    async def test_generation_timeout_handling(
        self, prompt_manager, json_parser, sample_generation_request
    ):
        """Test agent respects generation timeout"""
        # GIVEN: Agent with very short timeout
        slow_llm = AsyncMock()

        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(2)  # Slower than timeout

        slow_llm.generate.side_effect = slow_generate

        agent = QuestionGeneratorAgent(
            llm_service=slow_llm,
            prompt_manager=prompt_manager,
            json_parser=json_parser,
            config={"generation_timeout": 1, "max_retries": 1, "enable_fallback": False},
        )

        # WHEN: Processing request
        input_data = sample_generation_request.model_dump()
        result = await agent.process(input_data)

        # THEN: Should fail due to timeout
        assert result.success is False
        assert "timeout" in result.error.lower() or "failed" in result.error.lower()
