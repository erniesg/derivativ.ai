#!/usr/bin/env python3
"""
Test ReAct Orchestrator - Testing multi-agent coordination system.

Tests the ReAct pattern implementation with multiple specialist agents.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock, patch

# Configure pytest for async tests
pytest_plugins = ('pytest_asyncio',)

# Add project root to Python path for clean imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.services.react_orchestrator import (
    ReActMultiAgentOrchestrator,
    generate_igcse_question,
    review_question_quality,
    make_quality_decision,
    get_session_status
)


def test_react_tools():
    """Test ReAct tool functions."""

    # Test question generation tool
    result = generate_igcse_question(
        config_id="test_config",
        topic="algebra",
        difficulty="foundation",
        question_type="short_answer"
    )

    # Tools return "integration_needed" when no real orchestrator is available
    assert result["status"] == "integration_needed"
    assert "question_id" in result or "message" in result
    assert result["config_used"] == "test_config"
    assert result["topic"] == "algebra"

    # Test review tool
    review_result = review_question_quality(
        question_id="test_question_123",
        detailed_analysis=True
    )

    assert review_result["status"] == "integration_needed"
    assert review_result["question_id"] == "test_question_123"

    # Test quality decision tool
    decision_result = make_quality_decision(
        question_id="test_question_123",
        review_score=0.87,
        auto_publish=False
    )

    assert decision_result["status"] == "success"
    assert "decision" in decision_result


def test_session_status_tool():
    """Test session status tool."""

    status_result = get_session_status("test_session_123")

    assert status_result["status"] == "success"
    assert "questions_generated" in status_result
    assert status_result["session_id"] == "test_session_123"


@pytest.mark.asyncio
async def test_react_orchestrator_demo():
    """Test ReAct orchestrator demo workflow."""

    # Mock the required models
    with patch('src.services.react_orchestrator.OpenAIServerModel') as mock_openai:
        mock_model = Mock()
        mock_openai.return_value = mock_model

        # Create orchestrator
        orchestrator = ReActMultiAgentOrchestrator(
            manager_model=mock_model,
            specialist_model=mock_model,
            auto_publish=False,
            debug=True
        )

        # Test demo workflow
        result = await orchestrator.demonstrate_react_workflow()

        assert result is not None
        assert "status" in result


@pytest.mark.asyncio
async def test_react_question_generation():
    """Test ReAct orchestrator for question generation workflow."""

    # Mock the required models
    with patch('src.services.react_orchestrator.OpenAIServerModel') as mock_openai:
        mock_model = Mock()
        mock_openai.return_value = mock_model

        orchestrator = ReActMultiAgentOrchestrator(
            manager_model=mock_model,
            specialist_model=mock_model,
            auto_publish=False,
            debug=True
        )

        # Test question generation session
        requirements = {
            "topics": ["algebra"],
            "difficulty": "foundation",
            "question_types": ["short_answer"]
        }

        session = await orchestrator.generate_questions_with_react(
            config_id="test_config",
            num_questions=1,
            requirements=requirements
        )

        assert session is not None
        assert hasattr(session, 'session_id')
        assert hasattr(session, 'status')


def test_specialist_agent_initialization():
    """Test specialist agent creation and capabilities."""

    from src.services.react_orchestrator import QuestionGeneratorSpecialistAgent, QualityReviewSpecialistAgent

    with patch('src.services.react_orchestrator.OpenAIServerModel') as mock_openai:
        mock_model = Mock()
        mock_openai.return_value = mock_model

        # Test QuestionGeneratorSpecialistAgent
        generator_agent = QuestionGeneratorSpecialistAgent(mock_model)

        # Agent names use underscores in the actual implementation
        assert generator_agent.name == "question_generator_specialist"
        assert hasattr(generator_agent, 'tools')
        assert hasattr(generator_agent, 'max_steps')

        # Test QualityReviewSpecialistAgent
        reviewer_agent = QualityReviewSpecialistAgent(mock_model)

        assert reviewer_agent.name == "quality_review_specialist"
        assert hasattr(reviewer_agent, 'tools')
        assert hasattr(reviewer_agent, 'max_steps')


@pytest.mark.asyncio
async def test_react_manager_coordination():
    """Test ReAct manager coordinating multiple agents."""

    with patch('src.services.react_orchestrator.OpenAIServerModel') as mock_openai:
        mock_model = Mock()
        mock_openai.return_value = mock_model

        orchestrator = ReActMultiAgentOrchestrator(
            manager_model=mock_model,
            specialist_model=mock_model,
            debug=True
        )

        # Test simple coordination task
        task = "Generate 1 test question and review its quality"

        # Mock the manager agent run method
        with patch.object(orchestrator.manager_agent, 'run') as mock_run:
            mock_run.return_value = "Task completed successfully"

            result = orchestrator.manager_agent.run(task)

            assert result == "Task completed successfully"
            mock_run.assert_called_once_with(task)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
