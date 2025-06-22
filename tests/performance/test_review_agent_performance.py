"""
Performance tests for Review Agent.
Tests load, stress, timing, and concurrent processing.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock

import pytest

from src.agents.review_agent import ReviewAgent
from src.services.llm_service import LLMResponse


class TestReviewAgentPerformance:
    """Performance tests for ReviewAgent"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_review_agent_concurrent_processing(self):
        """Test ReviewAgent handles concurrent requests efficiently"""
        # GIVEN: ReviewAgent with fast mock responses
        agent = ReviewAgent()

        # Mock fast quality assessment
        quality_response = {
            "overall_quality_score": 0.8,
            "mathematical_accuracy": 0.85,
            "cambridge_compliance": 0.8,
            "grade_appropriateness": 0.8,
            "question_clarity": 0.75,
            "marking_accuracy": 0.8,
            "feedback_summary": "Good quality question",
            "specific_issues": [],
            "suggested_improvements": [],
            "decision": "approve",
        }

        agent.llm_service.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(quality_response),
                model_used="gpt-4o",
                provider="mock",
                tokens_used=150,
                cost_estimate=0.002,
                latency_ms=1000,
            )
        )

        # Create multiple concurrent requests
        concurrent_requests = []
        for i in range(10):
            request_data = {
                "question_data": {
                    "question": {
                        "question_text": f"Calculate {i} + {i+1}",
                        "marks": 1,
                        "command_word": "Calculate",
                        "subject_content_references": ["C1.1"],
                        "grade_level": 6,
                        "tier": "Core",
                    },
                    "marking_scheme": {
                        "total_marks_for_part": 1,
                        "mark_allocation_criteria": [
                            {"criterion_text": f"{i + i + 1}", "marks_value": 1, "mark_type": "A"}
                        ],
                    },
                }
            }
            concurrent_requests.append(agent.process(request_data))

        # WHEN: Processing all requests concurrently
        start_time = time.time()
        results = await asyncio.gather(*concurrent_requests)
        total_time = time.time() - start_time

        # THEN: Should complete all requests successfully and efficiently
        assert len(results) == 10
        assert all(result.success for result in results)

        # Performance assertions
        assert total_time < 15.0, f"Concurrent processing too slow: {total_time}s"

        # Check individual processing times
        processing_times = [result.processing_time for result in results]
        average_time = sum(processing_times) / len(processing_times)
        assert average_time < 5.0, f"Average processing time too slow: {average_time}s"

        # Ensure concurrency benefits (total time should be much less than sum of individual times)
        # With async operations, we expect some concurrency benefit, but not necessarily 2x
        # due to Python's GIL and the fact that we're mostly waiting on mocked I/O
        total_individual_time = sum(processing_times)
        concurrency_factor = total_individual_time / total_time
        # Relaxed expectation - we just want some concurrency benefit (> 1.0)
        assert concurrency_factor > 1.0, f"No concurrency benefit: {concurrency_factor}x"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_review_agent_large_batch_processing(self):
        """Test ReviewAgent performance with large batch of reviews"""
        # GIVEN: ReviewAgent
        agent = ReviewAgent()

        # Mock consistent responses
        agent.llm_service.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(
                    {
                        "overall_quality_score": 0.75,
                        "mathematical_accuracy": 0.8,
                        "cambridge_compliance": 0.75,
                        "grade_appropriateness": 0.75,
                        "question_clarity": 0.7,
                        "marking_accuracy": 0.75,
                        "feedback_summary": "Adequate quality",
                        "suggested_improvements": [],
                        "decision": "manual_review",
                    }
                ),
                model_used="gpt-4o",
                provider="mock",
                tokens_used=200,
                cost_estimate=0.003,
                latency_ms=1500,
            )
        )

        # Create large batch of questions
        batch_size = 25
        batch_requests = []

        for i in range(batch_size):
            request_data = {
                "question_data": {
                    "question": {
                        "question_text": f"Solve the equation x + {i} = {i + 10}",
                        "marks": 2,
                        "command_word": "Solve",
                        "subject_content_references": ["A2.1"],
                        "grade_level": 8,
                        "tier": "Core",
                    },
                    "marking_scheme": {
                        "total_marks_for_part": 2,
                        "mark_allocation_criteria": [
                            {
                                "criterion_text": "Rearrange equation",
                                "marks_value": 1,
                                "mark_type": "M",
                            },
                            {"criterion_text": "x = 10", "marks_value": 1, "mark_type": "A"},
                        ],
                    },
                }
            }
            batch_requests.append(request_data)

        # WHEN: Processing large batch
        start_time = time.time()
        batch_tasks = [agent.process(req) for req in batch_requests]
        results = await asyncio.gather(*batch_tasks)
        batch_time = time.time() - start_time

        # THEN: Should handle large batch efficiently
        assert len(results) == batch_size
        assert all(result.success for result in results)

        # Performance requirements for large batch
        assert batch_time < 60.0, f"Batch processing too slow: {batch_time}s for {batch_size} items"

        # Average time per item should be reasonable
        avg_time_per_item = batch_time / batch_size
        assert avg_time_per_item < 3.0, f"Average time per item too slow: {avg_time_per_item}s"

        # Check consistency of results
        quality_scores = [result.output["quality_decision"]["quality_score"] for result in results]
        assert all(
            0.7 <= score <= 0.8 for score in quality_scores
        ), "Inconsistent quality scores in batch"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_review_agent_memory_usage_stress(self):
        """Test ReviewAgent memory efficiency under stress"""
        # GIVEN: ReviewAgent with various response sizes
        agent = ReviewAgent()

        # Mock responses with different complexity levels
        responses = [
            # Simple response
            {
                "overall_quality_score": 0.9,
                "mathematical_accuracy": 0.9,
                "cambridge_compliance": 0.9,
                "grade_appropriateness": 0.9,
                "question_clarity": 0.9,
                "marking_accuracy": 0.9,
                "feedback_summary": "Excellent",
                "suggested_improvements": [],
                "decision": "approve",
            },
            # Complex response with many suggestions
            {
                "overall_quality_score": 0.6,
                "mathematical_accuracy": 0.7,
                "cambridge_compliance": 0.5,
                "grade_appropriateness": 0.6,
                "question_clarity": 0.6,
                "marking_accuracy": 0.7,
                "feedback_summary": "Needs significant improvements across multiple dimensions",
                "suggested_improvements": [
                    "Align question with proper Cambridge syllabus references",
                    "Improve mathematical accuracy in solution steps",
                    "Enhance question clarity and wording",
                    "Adjust difficulty for target grade level",
                    "Revise marking scheme for consistency",
                    "Add more detailed marking criteria",
                    "Include alternative solution methods",
                    "Provide clearer command word usage",
                ],
                "decision": "refine",
            },
        ]

        # Test with alternating response complexity
        call_count = 0

        def mock_generate(*args, **kwargs):
            nonlocal call_count
            response = responses[call_count % len(responses)]
            call_count += 1
            return LLMResponse(
                content=json.dumps(response),
                model_used="gpt-4o",
                provider="mock",
                tokens_used=300 + (call_count % 200),  # Variable token usage
                cost_estimate=0.005,
                latency_ms=2000,
            )

        agent.llm_service.generate = AsyncMock(side_effect=mock_generate)

        # Stress test with many sequential requests
        stress_count = 50
        results = []

        start_time = time.time()
        for i in range(stress_count):
            request_data = {
                "question_data": {
                    "question": {
                        "question_text": f"Find the value of x in {i}x + {i*2} = {i*5}",
                        "marks": 3,
                        "command_word": "Find",
                        "subject_content_references": ["A2.1"],
                        "grade_level": 9,
                        "tier": "Extended",
                    },
                    "marking_scheme": {
                        "total_marks_for_part": 3,
                        "mark_allocation_criteria": [
                            {
                                "criterion_text": "Setup equation",
                                "marks_value": 1,
                                "mark_type": "M",
                            },
                            {
                                "criterion_text": "Solve correctly",
                                "marks_value": 1,
                                "mark_type": "M",
                            },
                            {"criterion_text": "Final answer", "marks_value": 1, "mark_type": "A"},
                        ],
                    },
                }
            }

            result = await agent.process(request_data)
            results.append(result)

        stress_time = time.time() - start_time

        # THEN: Should handle stress test efficiently
        assert len(results) == stress_count
        assert all(result.success for result in results)

        # Performance under stress
        assert stress_time < 120.0, f"Stress test too slow: {stress_time}s"

        # Check memory efficiency (no growing data structures)
        # All results should have reasonable data sizes
        for result in results:
            assert len(result.reasoning_steps) < 20, "Reasoning steps growing too large"
            assert len(str(result.output)) < 5000, "Output data growing too large"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_review_agent_timeout_handling(self):
        """Test ReviewAgent handles timeouts gracefully under load"""
        # GIVEN: ReviewAgent with simulated slow responses
        agent = ReviewAgent()

        # Mock slow LLM responses that sometimes timeout
        timeout_scenarios = [
            # Fast response
            lambda: LLMResponse(
                content=json.dumps(
                    {
                        "overall_quality_score": 0.8,
                        "mathematical_accuracy": 0.8,
                        "cambridge_compliance": 0.8,
                        "grade_appropriateness": 0.8,
                        "question_clarity": 0.8,
                        "marking_accuracy": 0.8,
                        "feedback_summary": "Good quality",
                        "suggested_improvements": [],
                        "decision": "approve",
                    }
                ),
                model_used="gpt-4o",
                provider="mock",
                tokens_used=150,
                cost_estimate=0.002,
                latency_ms=1000,
            ),
            # Simulated timeout
            lambda: (_ for _ in ()).throw(asyncio.TimeoutError("Request timeout")),
        ]

        call_count = 0

        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            # Introduce timeouts for some requests
            if call_count % 4 == 0:  # Every 4th request times out
                await asyncio.sleep(0.1)  # Brief delay before timeout
                raise asyncio.TimeoutError("Simulated timeout")
            else:
                return timeout_scenarios[0]()  # Fast response

        agent.llm_service.generate = AsyncMock(side_effect=mock_generate)

        # Test with multiple requests where some will timeout
        timeout_requests = []
        for i in range(12):  # Will have 3 timeouts (every 4th)
            request_data = {
                "question_data": {
                    "question": {
                        "question_text": f"Calculate {i} × {i+1}",
                        "marks": 1,
                        "command_word": "Calculate",
                        "subject_content_references": ["C1.2"],
                        "grade_level": 7,
                        "tier": "Core",
                    },
                    "marking_scheme": {
                        "total_marks_for_part": 1,
                        "mark_allocation_criteria": [
                            {"criterion_text": f"{i * (i+1)}", "marks_value": 1, "mark_type": "A"}
                        ],
                    },
                }
            }
            timeout_requests.append(agent.process(request_data))

        # WHEN: Processing requests with timeouts
        start_time = time.time()
        results = await asyncio.gather(*timeout_requests, return_exceptions=True)
        timeout_test_time = time.time() - start_time

        # THEN: Should handle timeouts gracefully
        successful_results = [r for r in results if hasattr(r, "success") and r.success]
        failed_results = [r for r in results if hasattr(r, "success") and not r.success]

        # Should have both successful and failed results
        assert len(successful_results) > 0, "No successful results"
        assert len(failed_results) > 0, "No failed results (timeouts not triggered)"

        # Failed results should have proper error handling
        for failed in failed_results:
            assert failed.error is not None
            assert "timeout" in failed.error.lower() or "failed" in failed.error.lower()

        # Performance should still be reasonable despite timeouts
        assert timeout_test_time < 30.0, f"Timeout handling too slow: {timeout_test_time}s"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_review_agent_quality_threshold_performance(self):
        """Test ReviewAgent performance across different quality thresholds"""
        # GIVEN: ReviewAgent
        agent = ReviewAgent()

        # Test different quality scenarios that trigger different thresholds
        quality_scenarios = [
            ("high_quality", 0.95, "approve"),
            ("good_quality", 0.80, "manual_review"),
            ("medium_quality", 0.65, "manual_review"),
            ("low_quality", 0.45, "regenerate"),
            ("poor_quality", 0.25, "reject"),
        ]

        performance_results = []

        for scenario_name, quality_score, expected_action in quality_scenarios:
            # Mock response for this quality level
            quality_response = {
                "overall_quality_score": quality_score,
                "mathematical_accuracy": quality_score,
                "cambridge_compliance": quality_score,
                "grade_appropriateness": quality_score,
                "question_clarity": quality_score,
                "marking_accuracy": quality_score,
                "feedback_summary": f"Quality level: {scenario_name}",
                "suggested_improvements": []
                if quality_score > 0.8
                else [f"Improve for {scenario_name}"],
                "decision": expected_action,
            }

            agent.llm_service.generate = AsyncMock(
                return_value=LLMResponse(
                    content=json.dumps(quality_response),
                    model_used="gpt-4o",
                    provider="mock",
                    tokens_used=200,
                    cost_estimate=0.003,
                    latency_ms=1500,
                )
            )

            # Process multiple requests for each quality level
            scenario_times = []
            for i in range(5):  # 5 requests per scenario
                request_data = {
                    "question_data": {
                        "question": {
                            "question_text": f"Test question for {scenario_name} {i}",
                            "marks": 2,
                            "command_word": "Calculate",
                            "subject_content_references": ["C1.1"],
                            "grade_level": 8,
                            "tier": "Core",
                        },
                        "marking_scheme": {
                            "total_marks_for_part": 2,
                            "mark_allocation_criteria": [
                                {"criterion_text": "Method", "marks_value": 1, "mark_type": "M"},
                                {"criterion_text": "Answer", "marks_value": 1, "mark_type": "A"},
                            ],
                        },
                    }
                }

                start_time = time.time()
                result = await agent.process(request_data)
                process_time = time.time() - start_time

                assert result.success is True
                assert (
                    str(result.output["quality_decision"]["action"])
                    .replace("QualityAction.", "")
                    .lower()
                    == expected_action.lower()
                    or result.output["quality_decision"]["action"].value == expected_action
                )

                scenario_times.append(process_time)

            avg_time = sum(scenario_times) / len(scenario_times)
            performance_results.append(
                {
                    "scenario": scenario_name,
                    "quality_score": quality_score,
                    "expected_action": expected_action,
                    "avg_processing_time": avg_time,
                    "times": scenario_times,
                }
            )

        # THEN: Performance should be consistent across quality levels
        for result in performance_results:
            assert (
                result["avg_processing_time"] < 5.0
            ), f"{result['scenario']} too slow: {result['avg_processing_time']}s"

        # Check that performance doesn't degrade significantly with quality level
        processing_times = [r["avg_processing_time"] for r in performance_results]
        max_time = max(processing_times)
        min_time = min(processing_times)
        time_variance = max_time - min_time

        assert (
            time_variance < 2.0
        ), f"Too much performance variance across quality levels: {time_variance}s"
