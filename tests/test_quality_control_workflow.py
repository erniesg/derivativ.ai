"""
Test Quality Control Workflow - End-to-end testing of the automated quality improvement loop.

Tests the complete workflow:
- Question generation
- Review and scoring
- Quality decisions (approve/refine/regenerate/reject)
- Refinement process
- Database persistence
"""

import pytest
import uuid
import json
from unittest.mock import Mock, MagicMock
from datetime import datetime

from src.services.quality_control_workflow import QualityControlWorkflow, QualityDecision
from src.agents.refinement_agent import RefinementAgent
from src.models.question_models import (
    CandidateQuestion, CommandWord, CalculatorPolicy, GenerationStatus,
    QuestionTaxonomy, SolutionAndMarkingScheme, SolverAlgorithm, AnswerSummary,
    MarkAllocationCriterion, SolverStep, LLMModel
)


@pytest.mark.asyncio
class TestQualityControlWorkflow:
    """Test cases for the Quality Control Workflow."""

    @pytest.fixture
    def mock_agents_and_services(self):
        """Create mock agents and services for testing."""
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

    @pytest.fixture
    def sample_question(self):
        """Create a sample candidate question for testing."""
        return CandidateQuestion(
            question_id_local=str(uuid.uuid4()),
            question_id_global=str(uuid.uuid4()),
            question_number_display="1",
            marks=3,
            command_word=CommandWord.CALCULATE,
            raw_text_content="Calculate the area of a circle with radius 5 cm.",
            formatted_text_latex=None,
            taxonomy=QuestionTaxonomy(
                topic_path=["Geometry", "Circles"],
                subject_content_references=["C1.1"],
                skill_tags=["area_calculation"],
                cognitive_level="ProceduralFluency",
                difficulty_estimate_0_to_1=0.6
            ),
            solution_and_marking_scheme=SolutionAndMarkingScheme(
                final_answers_summary=[
                    AnswerSummary(answer_text="78.54 cm²", value_numeric=78.54, unit="cm²")
                ],
                mark_allocation_criteria=[
                    MarkAllocationCriterion(
                        criterion_id="1",
                        criterion_text="Correct formula and calculation",
                        mark_code_display="M3",
                        marks_value=3.0,
                        mark_type_primary="M"
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
                        skill_applied_tag="area_formula"
                    ),
                    SolverStep(
                        step_number=2,
                        description_text="Substitute values",
                        mathematical_expression_latex="A = \\pi \\times 5^2 = 25\\pi",
                        skill_applied_tag="substitution"
                    )
                ]
            ),
            generation_id=uuid.uuid4(),
            seed_question_id=None,
            target_grade_input=8,
            llm_model_used_generation=LLMModel.GPT_4O.value,
            llm_model_used_marking_scheme=LLMModel.GPT_4O.value,
            llm_model_used_review=LLMModel.CLAUDE_4_SONNET.value,
            prompt_template_version_generation="v1.0",
            prompt_template_version_marking_scheme="v1.0",
            prompt_template_version_review="v1.0",
            generation_timestamp=datetime.utcnow(),
            status=GenerationStatus.CANDIDATE,
            reviewer_notes=None,
            confidence_score=None,
            validation_errors=[]
        )

    async def test_auto_approve_workflow(self, mock_agents_and_services, sample_question):
        """Test workflow with high-quality question that gets auto-approved."""

        # Setup mocks
        mocks = mock_agents_and_services
        mocks['review_agent'].review_question.return_value = (
            {
                'overall_score': 0.90,
                'clarity_score': 0.92,
                'difficulty_score': 0.88,
                'curriculum_alignment_score': 0.91,
                'mathematical_accuracy_score': 0.89
            },
            {'interaction_id': str(uuid.uuid4()), 'success': True}
        )

        # Create workflow
        workflow = QualityControlWorkflow(**mocks)

        # Process question
        session_id = str(uuid.uuid4())
        config = {"type": "geometry_basic"}
        result = await workflow.process_question(sample_question, session_id, config)

        # Verify results
        assert result['success'] == True
        assert result['final_decision'] == QualityDecision.AUTO_APPROVE
        assert result['manual_review_required'] == False
        assert result['total_iterations'] == 1

        # Verify database interactions
        mocks['database_manager'].save_candidate_question.assert_called_once()
        mocks['database_manager'].save_llm_interaction.assert_called_once()

    async def test_refinement_workflow(self, mock_agents_and_services, sample_question):
        """Test workflow with medium-quality question that gets refined and then approved."""

        # Setup mocks
        mocks = mock_agents_and_services

        # First review: needs refinement
        # Second review: approved after refinement
        mocks['review_agent'].review_question.side_effect = [
            (
                {
                    'overall_score': 0.65,  # Triggers refinement
                    'clarity_score': 0.60,
                    'difficulty_score': 0.70,
                    'curriculum_alignment_score': 0.65,
                    'mathematical_accuracy_score': 0.65
                },
                {'interaction_id': str(uuid.uuid4()), 'success': True}
            ),
            (
                {
                    'overall_score': 0.87,  # Approved after refinement
                    'clarity_score': 0.85,
                    'difficulty_score': 0.89,
                    'curriculum_alignment_score': 0.88,
                    'mathematical_accuracy_score': 0.86
                },
                {'interaction_id': str(uuid.uuid4()), 'success': True}
            )
        ]

        # Create refined question (copy of original with changes)
        refined_question = sample_question.model_copy()
        refined_question.question_id_local = str(uuid.uuid4())
        refined_question.question_id_global = str(uuid.uuid4())
        refined_question.raw_text_content = "Calculate the area of a circle with radius 5 cm. Give your answer to 2 decimal places."
        refined_question.reviewer_notes = f"Refined from {sample_question.question_id_local}"

        mocks['refinement_agent'].refine_question.return_value = (
            refined_question,
            {'interaction_id': str(uuid.uuid4()), 'success': True}
        )

        # Create workflow
        workflow = QualityControlWorkflow(**mocks)

        # Process question
        session_id = str(uuid.uuid4())
        config = {"type": "geometry_basic"}
        result = await workflow.process_question(sample_question, session_id, config)

        # Verify results
        assert result['success'] == True
        assert result['final_decision'] == QualityDecision.AUTO_APPROVE
        assert result['total_iterations'] == 2
        assert len(result['steps']) == 2

        # Verify refinement was attempted
        mocks['refinement_agent'].refine_question.assert_called_once()

        # Verify final question is the refined one
        assert result['approved_question'].question_id_local == refined_question.question_id_local

    async def test_manual_review_workflow(self, mock_agents_and_services, sample_question):
        """Test workflow with borderline question that requires manual review."""

        # Setup mocks
        mocks = mock_agents_and_services
        mocks['review_agent'].review_question.return_value = (
            {
                'overall_score': 0.75,  # In manual review range
                'clarity_score': 0.80,
                'difficulty_score': 0.70,
                'curriculum_alignment_score': 0.75,
                'mathematical_accuracy_score': 0.75
            },
            {'interaction_id': str(uuid.uuid4()), 'success': True}
        )

        # Create workflow
        workflow = QualityControlWorkflow(**mocks)

        # Process question
        session_id = str(uuid.uuid4())
        config = {"type": "geometry_basic"}
        result = await workflow.process_question(sample_question, session_id, config)

        # Verify results
        assert result['success'] == True
        assert result['final_decision'] == QualityDecision.MANUAL_REVIEW
        assert result['manual_review_required'] == True
        assert result['total_iterations'] == 1

        # Verify question is saved for manual review
        mocks['database_manager'].save_candidate_question.assert_called()

    async def test_regeneration_workflow(self, mock_agents_and_services, sample_question):
        """Test workflow with low-quality question that gets regenerated."""

        # Setup mocks
        mocks = mock_agents_and_services

        # First review: needs regeneration
        # Second review: approved after regeneration
        mocks['review_agent'].review_question.side_effect = [
            (
                {
                    'overall_score': 0.45,  # Triggers regeneration
                    'clarity_score': 0.40,
                    'difficulty_score': 0.50,
                    'curriculum_alignment_score': 0.45,
                    'mathematical_accuracy_score': 0.45
                },
                {'interaction_id': str(uuid.uuid4()), 'success': True}
            ),
            (
                {
                    'overall_score': 0.88,  # Approved after regeneration
                    'clarity_score': 0.90,
                    'difficulty_score': 0.86,
                    'curriculum_alignment_score': 0.88,
                    'mathematical_accuracy_score': 0.88
                },
                {'interaction_id': str(uuid.uuid4()), 'success': True}
            )
        ]

        # Create regenerated question
        regenerated_question = sample_question.model_copy()
        regenerated_question.question_id_local = str(uuid.uuid4())
        regenerated_question.question_id_global = str(uuid.uuid4())
        regenerated_question.raw_text_content = "Find the area of a circle with diameter 10 cm. Give your answer in terms of π."

        mocks['generator_agent'].generate_question.return_value = {
            'success': True,
            'question': regenerated_question
        }

        # Create workflow
        workflow = QualityControlWorkflow(**mocks)

        # Process question
        session_id = str(uuid.uuid4())
        config = {"type": "geometry_basic"}
        result = await workflow.process_question(sample_question, session_id, config)

        # Verify results
        assert result['success'] == True
        assert result['final_decision'] == QualityDecision.AUTO_APPROVE
        assert result['total_iterations'] == 2

        # Verify regeneration was attempted
        mocks['generator_agent'].generate_question.assert_called_once()

        # Verify final question is the regenerated one
        assert result['approved_question'].question_id_local == regenerated_question.question_id_local

    async def test_rejection_workflow(self, mock_agents_and_services, sample_question):
        """Test workflow with very low-quality question that gets rejected."""

        # Setup mocks
        mocks = mock_agents_and_services
        mocks['review_agent'].review_question.return_value = (
            {
                'overall_score': 0.25,  # Very low score, triggers rejection
                'clarity_score': 0.20,
                'difficulty_score': 0.30,
                'curriculum_alignment_score': 0.25,
                'mathematical_accuracy_score': 0.25
            },
            {'interaction_id': str(uuid.uuid4()), 'success': True}
        )

        # Create workflow
        workflow = QualityControlWorkflow(**mocks)

        # Process question
        session_id = str(uuid.uuid4())
        config = {"type": "geometry_basic"}
        result = await workflow.process_question(sample_question, session_id, config)

        # Verify results
        assert result['success'] == False
        assert result['final_decision'] == QualityDecision.REJECT
        assert result['total_iterations'] == 1

        # Verify rejection was logged
        mocks['database_manager'].save_error_log.assert_called_once()

    async def test_max_refinement_iterations(self, mock_agents_and_services, sample_question):
        """Test that workflow stops after max refinement iterations."""

        # Setup mocks
        mocks = mock_agents_and_services

        # Always return score that triggers refinement
        mocks['review_agent'].review_question.return_value = (
            {
                'overall_score': 0.65,  # Always triggers refinement
                'clarity_score': 0.60,
                'difficulty_score': 0.70,
                'curriculum_alignment_score': 0.65,
                'mathematical_accuracy_score': 0.65
            },
            {'interaction_id': str(uuid.uuid4()), 'success': True}
        )

        # Mock refinement to always return a question
        refined_question = sample_question
        refined_question.question_id_local = str(uuid.uuid4())
        mocks['refinement_agent'].refine_question.return_value = (
            refined_question,
            {'interaction_id': str(uuid.uuid4()), 'success': True}
        )

        # Create workflow with custom limits
        workflow = QualityControlWorkflow(**mocks)
        workflow.max_refinement_iterations = 2  # Lower limit for testing

        # Process question
        session_id = str(uuid.uuid4())
        config = {"type": "geometry_basic"}
        result = await workflow.process_question(sample_question, session_id, config)

        # Verify it stops after max iterations and goes to manual review
        assert result['final_decision'] == QualityDecision.MANUAL_REVIEW
        assert result['manual_review_required'] == True

        # Should have called refinement agent max_refinement_iterations times
        assert mocks['refinement_agent'].refine_question.call_count == 2

    async def test_custom_quality_thresholds(self, mock_agents_and_services, sample_question):
        """Test workflow with custom quality thresholds."""

        # Setup mocks
        mocks = mock_agents_and_services
        mocks['review_agent'].review_question.return_value = (
            {
                'overall_score': 0.80,
                'clarity_score': 0.80,
                'difficulty_score': 0.80,
                'curriculum_alignment_score': 0.80,
                'mathematical_accuracy_score': 0.80
            },
            {'interaction_id': str(uuid.uuid4()), 'success': True}
        )

        # Create workflow with custom thresholds
        custom_thresholds = {
            'auto_approve': 0.95,  # Very high threshold
            'manual_review': 0.85,
            'refine': 0.70,
            'regenerate': 0.50
        }
        workflow = QualityControlWorkflow(**mocks, quality_thresholds=custom_thresholds)

        # Process question
        session_id = str(uuid.uuid4())
        config = {"type": "geometry_basic"}
        result = await workflow.process_question(sample_question, session_id, config)

        # With custom thresholds, 0.80 should require manual review
        assert result['final_decision'] == QualityDecision.MANUAL_REVIEW
        assert result['manual_review_required'] == True


class TestRefinementAgent:
    """Test cases for the Refinement Agent."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        mock = Mock()
        mock.__str__ = Mock(return_value="mock-model")
        return mock

    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock config manager."""
        return Mock()

    @pytest.fixture
    def sample_question(self):
        """Create a sample question for refinement testing."""
        return CandidateQuestion(
            question_id_local=str(uuid.uuid4()),
            question_id_global=str(uuid.uuid4()),
            question_number_display="1",
            marks=3,
            command_word=CommandWord.CALCULATE,
            raw_text_content="Calculate area of circle radius 5",
            formatted_text_latex=None,
            taxonomy=QuestionTaxonomy(
                topic_path=["Geometry"],
                subject_content_references=["C1.1"],
                skill_tags=["area_calculation"]
            ),
            solution_and_marking_scheme=SolutionAndMarkingScheme(
                final_answers_summary=[
                    AnswerSummary(answer_text="78.54 cm²")
                ],
                mark_allocation_criteria=[
                    MarkAllocationCriterion(
                        criterion_id="1",
                        criterion_text="Correct calculation",
                        mark_code_display="M3",
                        marks_value=3.0
                    )
                ],
                total_marks_for_part=3
            ),
            solver_algorithm=SolverAlgorithm(steps=[]),
            generation_id=uuid.uuid4(),
            target_grade_input=8,
            llm_model_used_generation=LLMModel.GPT_4O.value,
            llm_model_used_marking_scheme=LLMModel.GPT_4O.value,
            prompt_template_version_generation="v1.0",
            prompt_template_version_marking_scheme="v1.0"
        )

    def test_successful_refinement(self, mock_model, mock_config_manager, sample_question):
        """Test successful question refinement."""

        # The refinement agent expects a complete structure with ALL required fields
        response_data = {
            "question_id_local": "Ref_Q1234",
            "question_id_global": "ref_original_567",
            "question_number_display": "Refined Question",
            "marks": 3,
            "command_word": "Calculate",
            "raw_text_content": "Calculate the area of a circle with radius 5 cm. Give your answer to 2 decimal places.",
            "formatted_text_latex": None,
            "taxonomy": {
                "topic_path": ["Geometry", "Circles"],
                "subject_content_references": ["C1.1"],
                "skill_tags": ["area_calculation", "precision"],
                "cognitive_level": "ProceduralFluency",
                "difficulty_estimate_0_to_1": 0.6
            },
            "solution_and_marking_scheme": {
                "final_answers_summary": [
                    {
                        "answer_text": "78.54 cm²",
                        "value_numeric": 78.54,
                        "unit": "cm²"
                    }
                ],
                "mark_allocation_criteria": [
                    {
                        "criterion_id": "ref_crit_1",
                        "criterion_text": "Correct use of area formula and calculation to 2 d.p.",
                        "mark_code_display": "M3",
                        "marks_value": 3.0,
                        "mark_type_primary": "M",
                        "qualifiers_and_notes": "Accept π × 25 = 78.54"
                    }
                ],
                "total_marks_for_part": 3
            },
            "solver_algorithm": {
                "steps": [
                    {
                        "step_number": 1,
                        "description_text": "Apply area formula",
                        "mathematical_expression_latex": "A = \\pi r^2",
                        "skill_applied_tag": "area_formula",
                        "justification_or_reasoning": "Use the standard formula for circle area"
                    },
                    {
                        "step_number": 2,
                        "description_text": "Substitute and calculate",
                        "mathematical_expression_latex": "A = \\pi \\times 5^2 = 25\\pi = 78.54",
                        "skill_applied_tag": "calculation",
                        "justification_or_reasoning": "Calculate to 2 decimal places"
                    }
                ]
            }
        }

        # Mock model response with properly escaped JSON in code block format
        mock_response = Mock()
        mock_response.content = f"```json\n{json.dumps(response_data)}\n```"
        mock_model.chat.return_value = mock_response

        # Create refinement agent
        agent = RefinementAgent(mock_model, mock_config_manager)

        # Mock prompt loader
        agent.prompt_loader.format_refinement_prompt = Mock(return_value="mocked prompt")

        # Create review feedback
        review_feedback = {
            'overall_score': 0.65,
            'clarity_score': 0.60,
            'difficulty_score': 0.70,
            'curriculum_alignment_score': 0.65,
            'mathematical_accuracy_score': 0.65
        }

        # Refine question
        interaction_id = str(uuid.uuid4())
        refined_question, interaction_data = agent.refine_question(
            sample_question, review_feedback, interaction_id
        )

        # Verify results
        assert refined_question is not None
        assert refined_question.raw_text_content == "Calculate the area of a circle with radius 5 cm. Give your answer to 2 decimal places."
        assert refined_question.reviewer_notes == f"Refined from {sample_question.question_id_local}"
        assert refined_question.question_id_local == "Ref_Q1234"
        assert refined_question.marks == 3
        assert len(refined_question.solution_and_marking_scheme.final_answers_summary) == 1
        assert len(refined_question.solver_algorithm.steps) == 2
        assert interaction_data['success'] == True
        assert interaction_data['agent_type'] == 'refinement'

    def test_refinement_with_fallback(self, mock_model, mock_config_manager, sample_question):
        """Test refinement that fails initially but succeeds with fallback."""

        # Mock model responses - first fails, second succeeds
        responses = [
            "Invalid response without JSON",  # First attempt fails
            '''{
                "question_text": "Improved question via fallback",
                "answer": "78.54 cm²",
                "working": "Improved working",
                "marks": 3,
                "topic": "Geometry",
                "difficulty": "Foundation"
            }'''  # Fallback succeeds
        ]

        mock_model.chat.side_effect = [Mock(content=resp) for resp in responses]

        # Create refinement agent
        agent = RefinementAgent(mock_model, mock_config_manager)

        # Mock prompt loader
        agent.prompt_loader.format_refinement_prompt = Mock(return_value="mocked prompt")

        # Create review feedback
        review_feedback = {
            'overall_score': 0.65,
            'clarity_score': 0.60
        }

        # Refine question
        interaction_id = str(uuid.uuid4())
        refined_question, interaction_data = agent.refine_question(
            sample_question, review_feedback, interaction_id
        )

        # Verify fallback was used
        assert refined_question is not None
        assert refined_question.raw_text_content == "Improved question via fallback"
        assert interaction_data['success'] == True
        assert interaction_data['attempt_number'] == 2  # Used fallback

    def test_refinement_complete_failure(self, mock_model, mock_config_manager, sample_question):
        """Test refinement that fails completely."""

        # Mock model to always return invalid responses
        mock_model.chat.return_value = Mock(content="Invalid response")

        # Create refinement agent
        agent = RefinementAgent(mock_model, mock_config_manager)

        # Mock prompt loader
        agent.prompt_loader.format_refinement_prompt = Mock(return_value="mocked prompt")

        # Create review feedback
        review_feedback = {'overall_score': 0.65}

        # Refine question
        interaction_id = str(uuid.uuid4())
        refined_question, interaction_data = agent.refine_question(
            sample_question, review_feedback, interaction_id
        )

        # Verify failure
        assert refined_question is None
        assert interaction_data['success'] == False


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v"])
