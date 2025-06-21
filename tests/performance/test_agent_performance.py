"""
Performance tests for agents.
Tests load, stress, timing, and concurrency scenarios.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock

import pytest

from src.agents.marker_agent import MarkerAgent
from src.agents.question_generator import QuestionGeneratorAgent
from src.services.llm_service import LLMResponse, LLMTimeoutError


class TestAgentPerformance:
    """Performance and load tests for agents"""

    @pytest.mark.asyncio
    async def test_concurrent_question_generation(self):
        """Test multiple concurrent question generation requests"""
        # GIVEN: Agent with fast mocked LLM responses (to test agent logic performance)
        agent = QuestionGeneratorAgent()

        # Mock the LLM service to return valid question JSON quickly
        question_json = {
            "question_text": "Calculate the value of 3x + 2 when x = 5",
            "marks": 2,
            "command_word": "Calculate",
            "solution_steps": ["Substitute x = 5", "Calculate 3(5) + 2", "Simplify to get 17"],
            "final_answer": "17",
        }

        agent.llm_service.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(question_json),
                model_used="gpt-4o",
                provider="mock",
                tokens_used=150,
                cost_estimate=0.002,
                latency_ms=100,  # Fast response to test agent concurrency
            )
        )

        request_data = {"topic": "algebra", "marks": 2, "tier": "Core"}

        # WHEN: Processing multiple requests concurrently
        tasks = [agent.process(request_data) for _ in range(5)]
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time

        # THEN: All requests should succeed within reasonable time
        assert len(results) == 5
        assert all(result.success for result in results)
        assert duration < 30  # Should complete within 30 seconds

    @pytest.mark.asyncio
    async def test_concurrent_marking_generation(self):
        """Test multiple concurrent marking scheme generation requests"""
        # GIVEN: Agent with mocked LLM service
        agent = MarkerAgent()

        # Mock the LLM service to return valid marking scheme JSON
        marking_scheme_json = {
            "total_marks": 2,
            "mark_allocation_criteria": [
                {
                    "criterion_text": "Set up percentage calculation",
                    "marks_value": 1,
                    "mark_type": "M",
                },
                {"criterion_text": "Correct answer: 36", "marks_value": 1, "mark_type": "A"},
            ],
            "final_answers": [{"answer_text": "36", "value_numeric": 36}],
        }

        agent.llm_service.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(marking_scheme_json),
                model_used="gpt-4o",
                provider="mock",
                tokens_used=200,
                cost_estimate=0.003,
                latency_ms=1500,
            )
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
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time

        # THEN: All requests should succeed within reasonable time
        assert len(results) == 5
        assert all(result.success for result in results)
        assert duration < 30  # Should complete within 30 seconds

    @pytest.mark.asyncio
    async def test_question_generation_timeout_handling(self):
        """Test agent handles timeout scenarios gracefully"""
        # GIVEN: Agent with slow LLM service
        agent = QuestionGeneratorAgent()

        # Mock a slow LLM service that times out
        slow_llm = AsyncMock()

        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(2)  # Slower than timeout

        slow_llm.generate.side_effect = slow_generate

        agent.llm_service = slow_llm
        agent.agent_config["generation_timeout"] = 1  # Very short timeout
        agent.agent_config["max_retries"] = 1
        agent.agent_config["enable_fallback"] = False

        # WHEN: Processing request
        request = {"topic": "algebra", "marks": 2, "tier": "Core"}
        start_time = time.time()
        result = await agent.process(request)
        duration = time.time() - start_time

        # THEN: Should fail quickly due to timeout, not hang
        assert result.success is False
        assert duration < 10  # Should fail quickly, not hang
        assert "timeout" in result.error.lower() or "failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_marking_generation_timeout_handling(self):
        """Test marker agent handles timeout scenarios gracefully"""
        # GIVEN: Agent with slow LLM service
        agent = MarkerAgent()

        # Mock a slow LLM service
        slow_llm = AsyncMock()

        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(2)

        slow_llm.generate.side_effect = slow_generate

        agent.llm_service = slow_llm
        agent.agent_config["generation_timeout"] = 1
        agent.agent_config["max_retries"] = 1
        agent.agent_config["enable_fallback"] = False

        # WHEN: Processing request
        request = {
            "question": {"question_text": "Calculate 2 + 3", "marks": 2},
            "config": {"topic": "arithmetic", "marks": 2, "tier": "Core"},
        }
        start_time = time.time()
        result = await agent.process(request)
        duration = time.time() - start_time

        # THEN: Should fail quickly
        assert result.success is False
        assert duration < 10
        assert "timeout" in result.error.lower() or "failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_retry_logic_performance(self):
        """Test that retry logic doesn't cause excessive delays"""
        # GIVEN: Agent with LLM that fails first few times then succeeds
        agent = QuestionGeneratorAgent()

        call_count = 0

        async def flaky_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LLMTimeoutError("Simulated timeout")
            # Success on 3rd try
            return LLMResponse(
                content='{"question_text": "Test question", "marks": 2, "command_word": "Calculate"}',
                model_used="gpt-4o",
                provider="mock",
                tokens_used=50,
                cost_estimate=0.001,
                latency_ms=1000,
            )

        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = flaky_generate
        agent.llm_service = mock_llm
        agent.agent_config["max_retries"] = 3

        # WHEN: Processing request with retries
        request = {"topic": "algebra", "marks": 2, "tier": "Core"}
        start_time = time.time()
        result = await agent.process(request)
        duration = time.time() - start_time

        # THEN: Should succeed but not take too long
        assert result.success is True
        assert duration < 15  # Should complete within reasonable time even with retries
        assert call_count == 3  # Should have retried as expected

    @pytest.mark.asyncio
    async def test_large_batch_processing(self):
        """Test processing a larger batch of requests"""
        # GIVEN: Question agent with mocked LLM service
        question_agent = QuestionGeneratorAgent()

        # Mock the LLM service to return valid question JSON
        question_json = {
            "question_text": "Calculate the value of 2x + 3 when x = 4",
            "marks": 2,
            "command_word": "Calculate",
            "solution_steps": ["Substitute x = 4", "Calculate 2(4) + 3", "Simplify to get 11"],
            "final_answer": "11",
        }

        question_agent.llm_service.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(question_json),
                model_used="gpt-4o",
                provider="mock",
                tokens_used=150,
                cost_estimate=0.002,
                latency_ms=1000,
            )
        )

        topics = ["algebra", "geometry", "statistics", "trigonometry", "calculus"]
        batch_size = 10

        # WHEN: Processing large batch
        start_time = time.time()

        question_tasks = []
        for i in range(batch_size):
            topic = topics[i % len(topics)]
            request = {"topic": topic, "marks": 2, "tier": "Core"}
            question_tasks.append(question_agent.process(request))

        question_results = await asyncio.gather(*question_tasks)
        duration = time.time() - start_time

        # THEN: Should handle batch efficiently
        assert len(question_results) == batch_size
        successful_questions = [r for r in question_results if r.success]
        assert len(successful_questions) >= batch_size * 0.8  # At least 80% success rate
        assert duration < 60  # Should complete batch within 1 minute
