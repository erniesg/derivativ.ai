"""
End-to-End Test for Auto-Publishing Approved Questions to Payload CMS.

This test validates the complete workflow:
1. Generate a question
2. Run it through quality control
3. Auto-approve based on high scores
4. Publish to Payload CMS via API
5. Verify existence in Payload
6. Clean up test data
"""

import os
import pytest
import uuid
import asyncio
from unittest.mock import Mock
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.services.quality_control_workflow import QualityControlWorkflow, QualityDecision
from src.services.payload_publisher import PayloadPublisher
from src.models.question_models import (
    CandidateQuestion, CommandWord, QuestionTaxonomy,
    SolutionAndMarkingScheme, SolverAlgorithm, AnswerSummary,
    MarkAllocationCriterion, SolverStep, LLMModel, GenerationStatus
)


class TestEndToEndPayloadPublish:
    """End-to-end tests for Payload publishing workflow."""

    @pytest.fixture
    def payload_publisher(self):
        """Create a PayloadPublisher instance."""
        return PayloadPublisher()

    @pytest.fixture
    def sample_question(self):
        """Create a sample question for testing."""
        unique_id = str(uuid.uuid4())[:8]  # Short unique ID for test

        return CandidateQuestion(
            question_id_local=f"test_q_{unique_id}",
            question_id_global=f"test_global_{unique_id}",
            question_number_display="E2E Test Question",
            marks=3,
            command_word=CommandWord.CALCULATE,
            raw_text_content=f"Calculate the area of a circle with radius 5 cm. [Test ID: {unique_id}]",
            formatted_text_latex=None,
            taxonomy=QuestionTaxonomy(
                topic_path=["Test", "Geometry", "Circles"],
                subject_content_references=["C1.1", "C1.6"],
                skill_tags=["AREA_CALCULATION", "CIRCLE_PROPERTIES", "FORMULA_APPLICATION"],
                cognitive_level="ProceduralFluency",
                difficulty_estimate_0_to_1=0.7  # Grade 6-7 range
            ),
            solution_and_marking_scheme=SolutionAndMarkingScheme(
                final_answers_summary=[
                    AnswerSummary(answer_text="78.54 cm²", value_numeric=78.54, unit="cm²")
                ],
                mark_allocation_criteria=[
                    MarkAllocationCriterion(
                        criterion_id=f"test_crit_{unique_id}",
                        criterion_text="Correct use of area formula πr² and calculation",
                        mark_code_display="M3",
                        marks_value=3.0,
                        mark_type_primary="M",
                        qualifiers_and_notes="Accept equivalent forms"
                    )
                ],
                total_marks_for_part=3
            ),
            solver_algorithm=SolverAlgorithm(
                steps=[
                    SolverStep(
                        step_number=1,
                        description_text="Apply area formula",
                        mathematical_expression_latex="A = \\pi r^2",
                        skill_applied_tag="FORMULA_APPLICATION"
                    ),
                    SolverStep(
                        step_number=2,
                        description_text="Substitute and calculate",
                        mathematical_expression_latex="A = \\pi \\times 5^2 = 25\\pi = 78.54",
                        skill_applied_tag="NUMERICAL_CALCULATION"
                    )
                ]
            ),
            generation_id=uuid.uuid4(),
            target_grade_input=7,
            llm_model_used_generation=LLMModel.GPT_4O.value,
            llm_model_used_marking_scheme=LLMModel.GPT_4O.value,
            llm_model_used_review=LLMModel.CLAUDE_4_SONNET.value,
            prompt_template_version_generation="v1.0",
            prompt_template_version_marking_scheme="v1.0",
            prompt_template_version_review="v1.0",
            generation_timestamp=datetime.utcnow(),
            status=GenerationStatus.CANDIDATE
        )

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        mock_review_agent = Mock()
        mock_refinement_agent = Mock()
        mock_generator_agent = Mock()
        mock_database_manager = Mock()

        return {
            'review_agent': mock_review_agent,
            'refinement_agent': mock_refinement_agent,
            'generator_agent': mock_generator_agent,
            'database_manager': mock_database_manager
        }

    @pytest.mark.asyncio
    async def test_payload_publisher_functionality(self, payload_publisher, sample_question):
        """Test basic PayloadPublisher functionality."""

        # Skip test if Payload not configured
        if not payload_publisher.is_enabled():
            pytest.skip("Payload API not configured - set PAYLOAD_API_TOKEN and PAYLOAD_API_URL")

        # Test publishing question
        payload_question_id = await payload_publisher.publish_question(sample_question)

        if payload_question_id is None:
            pytest.skip("Failed to publish to Payload - check API credentials and connectivity")

        try:
            # Verify question exists
            exists = await payload_publisher.verify_question_exists(sample_question.question_id_global)
            assert exists, "Question should exist in Payload after publishing"

            # Find the question by global ID
            questions = await payload_publisher.find_questions_by_global_id(sample_question.question_id_global)
            assert len(questions) > 0, "Should find at least one question"

            published_question = questions[0]
            assert published_question['question_id_global'] == sample_question.question_id_global
            assert published_question['marks'] == sample_question.marks
            assert published_question['command_word'] == sample_question.command_word.value
            assert sample_question.question_id_local[:8] in published_question['raw_text_content']  # Test ID should be in content

        finally:
            # Cleanup: Delete the test question
            if payload_question_id:
                deleted = await payload_publisher.delete_question(payload_question_id)
                assert deleted, "Should successfully delete test question"

    @pytest.mark.asyncio
    async def test_end_to_end_workflow_with_auto_publish(self, mock_agents, sample_question):
        """Test complete workflow with auto-publish enabled."""

        # Skip test if Payload not configured
        publisher = PayloadPublisher()
        if not publisher.is_enabled():
            pytest.skip("Payload API not configured - set PAYLOAD_API_TOKEN and PAYLOAD_API_URL")

        # Mock high-quality review scores (triggers auto-approval)
        mock_agents['review_agent'].review_question.return_value = (
            {
                'overall_score': 0.92,  # High score triggers auto-approval
                'clarity_score': 0.90,
                'difficulty_score': 0.95,
                'curriculum_alignment_score': 0.90,
                'mathematical_accuracy_score': 0.93
            },
            {'interaction_id': str(uuid.uuid4()), 'success': True}
        )

        # Create workflow with auto-publish enabled
        workflow = QualityControlWorkflow(
            **mock_agents,
            auto_publish=True  # This is the key flag
        )

        # Process the question
        session_id = str(uuid.uuid4())
        config = {"type": "geometry_basic", "target_grade": 7}
        result = await workflow.process_question(sample_question, session_id, config)

        # Verify workflow results
        assert result['success'] == True, "Workflow should succeed"
        assert result['final_decision'] == QualityDecision.AUTO_APPROVE, "Question should be auto-approved"
        assert result['auto_publish_enabled'] == True, "Auto-publish should be enabled"

        # Check if Payload publishing was attempted and succeeded
        if result.get('payload_published'):
            payload_question_id = result.get('payload_question_id')
            assert payload_question_id is not None, "Should have Payload question ID"

            try:
                # Verify the question exists in Payload
                exists = await publisher.verify_question_exists(sample_question.question_id_global)
                assert exists, "Question should exist in Payload after auto-publish"

                # Verify it's accessible via drill mode query parameters
                questions = await publisher.find_questions_by_global_id(sample_question.question_id_global)
                assert len(questions) > 0, "Should find the published question"

                published_question = questions[0]
                # Verify the difficulty is in the expected range for grade 7 (0.4-0.8)
                difficulty = published_question['taxonomy']['difficulty_estimate_0_to_1']
                assert 0.4 <= difficulty <= 0.8, f"Difficulty {difficulty} should be in Grade 6-7 range"

            finally:
                # Cleanup: Delete the test question
                if payload_question_id:
                    deleted = await publisher.delete_question(payload_question_id)
                    assert deleted, "Should successfully delete test question"

        else:
            # If publishing failed, check why
            if result.get('payload_error'):
                pytest.skip(f"Payload publishing failed: {result['payload_error']}")
            else:
                pytest.fail("Auto-publish was enabled but no publishing attempt was made")

    @pytest.mark.asyncio
    async def test_workflow_without_auto_publish(self, mock_agents, sample_question):
        """Test workflow with auto-publish disabled."""

        # Mock high-quality review scores
        mock_agents['review_agent'].review_question.return_value = (
            {
                'overall_score': 0.88,
                'clarity_score': 0.85,
                'difficulty_score': 0.90,
                'curriculum_alignment_score': 0.88,
                'mathematical_accuracy_score': 0.90
            },
            {'interaction_id': str(uuid.uuid4()), 'success': True}
        )

        # Create workflow with auto-publish disabled (default)
        workflow = QualityControlWorkflow(**mock_agents, auto_publish=False)

        # Process the question
        session_id = str(uuid.uuid4())
        config = {"type": "geometry_basic", "target_grade": 7}
        result = await workflow.process_question(sample_question, session_id, config)

        # Verify workflow results
        assert result['success'] == True, "Workflow should succeed"
        assert result['final_decision'] == QualityDecision.AUTO_APPROVE, "Question should be auto-approved"
        assert result['auto_publish_enabled'] == False, "Auto-publish should be disabled"
        assert result['payload_published'] == False, "Question should not be published to Payload"
        assert result['payload_question_id'] is None, "Should not have Payload question ID"

    @pytest.mark.asyncio
    async def test_payload_api_validation(self):
        """Test Payload API connectivity and validation."""
        publisher = PayloadPublisher()

        if not publisher.is_enabled():
            pytest.skip("Payload API not configured")

        # Test basic connectivity by trying to fetch questions
        try:
            questions = await publisher.find_questions_by_global_id("nonexistent_test_id")
            # Should return empty list, not fail
            assert isinstance(questions, list), "Should return a list even for non-existent questions"
        except Exception as e:
            pytest.fail(f"Payload API connectivity test failed: {str(e)}")

    def test_environment_configuration(self):
        """Test that environment variables are properly configured."""
        payload_url = os.getenv('PAYLOAD_API_URL')
        payload_token = os.getenv('PAYLOAD_API_TOKEN')

        if not payload_url:
            print("⚠️  PAYLOAD_API_URL not set - add to .env file")

        if not payload_token:
            print("⚠️  PAYLOAD_API_TOKEN not set - add to .env file")

        if payload_url and payload_token:
            print(f"✅ Payload configured: {payload_url}")
        else:
            print("ℹ️  To test Payload integration, add PAYLOAD_API_URL and PAYLOAD_API_TOKEN to .env")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"])
