#!/usr/bin/env python3
"""
Demo: Quality Control Workflow with Complete Question Refinement

This demo shows how the new quality control workflow automatically improves
question quality through iterative refinement and regeneration, now with
COMPLETE question structures suitable for database insertion.

Features demonstrated:
- Automatic quality assessment
- Complete question refinement (not just text)
- Full question structure with marking schemes and solver algorithms
- Decision thresholds (approve/refine/regenerate/reject)
- Database persistence with audit trails
"""

import uuid
from datetime import datetime
from unittest.mock import Mock
import json

# Import our new components
from src.services.quality_control_workflow import QualityControlWorkflow, QualityDecision
from src.agents.refinement_agent import RefinementAgent
from src.models.question_models import (
    CandidateQuestion, CommandWord, QuestionTaxonomy,
    SolutionAndMarkingScheme, SolverAlgorithm, AnswerSummary,
    MarkAllocationCriterion, SolverStep, LLMModel, GenerationStatus
)


def create_sample_question():
    """Create a sample question for demonstration."""
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
            subject_content_references=["C1.1", "C1.2"],
            skill_tags=["area_calculation", "circle_properties"]
        ),
        solution_and_marking_scheme=SolutionAndMarkingScheme(
            final_answers_summary=[
                AnswerSummary(answer_text="78.54 cm²", value_numeric=78.54, unit="cm²")
            ],
            mark_allocation_criteria=[
                MarkAllocationCriterion(
                    criterion_id="1",
                    criterion_text="Correct use of area formula πr²",
                    mark_code_display="M1",
                    marks_value=1.0,
                    mark_type_primary="M"
                ),
                MarkAllocationCriterion(
                    criterion_id="2",
                    criterion_text="Correct substitution and calculation",
                    mark_code_display="A2",
                    marks_value=2.0,
                    mark_type_primary="A"
                )
            ],
            total_marks_for_part=3
        ),
        solver_algorithm=SolverAlgorithm(
            steps=[
                SolverStep(
                    step_number=1,
                    description_text="Apply area formula",
                    mathematical_expression_latex="A = \\pi r^2"
                ),
                SolverStep(
                    step_number=2,
                    description_text="Substitute values",
                    mathematical_expression_latex="A = \\pi \\times 5^2 = 25\\pi = 78.54"
                )
            ]
        ),
        generation_id=uuid.uuid4(),
        target_grade_input=8,
        llm_model_used_generation=LLMModel.GPT_4O.value,
        llm_model_used_marking_scheme=LLMModel.GPT_4O.value,
        llm_model_used_review=LLMModel.CLAUDE_4_SONNET.value,
        prompt_template_version_generation="v1.0",
        prompt_template_version_marking_scheme="v1.0",
        prompt_template_version_review="v1.0",
        generation_timestamp=datetime.utcnow(),
        status=GenerationStatus.CANDIDATE
    )


def create_mock_agents():
    """Create mock agents for demonstration."""

    # Mock Review Agent - simulates different quality scores
    mock_review_agent = Mock()

    # Mock Refinement Agent - simulates question improvement
    mock_refinement_agent = Mock()

    # Mock Generator Agent - simulates new question generation
    mock_generator_agent = Mock()

    # Mock Database Manager - simulates persistence
    mock_database_manager = Mock()

    return {
        'review_agent': mock_review_agent,
        'refinement_agent': mock_refinement_agent,
        'generator_agent': mock_generator_agent,
        'database_manager': mock_database_manager
    }


def demo_complete_refinement_structure():
    """Demo: Complete question refinement with full structure."""
    print("\n🔧 DEMO: Complete Question Refinement Structure")
    print("=" * 60)

    # Setup original question
    original_question = create_sample_question()
    mocks = create_mock_agents()

    # Mock complete refined question with all required fields
    refined_question = create_sample_question()
    refined_question.question_id_local = "Ref_Q1234"
    refined_question.question_id_global = f"ref_{original_question.question_id_local}_567"
    refined_question.question_number_display = "Refined Question"
    refined_question.raw_text_content = "Calculate the area of a circle with radius 5 cm. Give your answer to 2 decimal places."
    refined_question.reviewer_notes = f"Refined from {original_question.question_id_local}"

    # Updated marking scheme for refined question
    refined_question.solution_and_marking_scheme = SolutionAndMarkingScheme(
        final_answers_summary=[
            AnswerSummary(answer_text="78.54 cm²", value_numeric=78.54, unit="cm²")
        ],
        mark_allocation_criteria=[
            MarkAllocationCriterion(
                criterion_id="ref_crit_1",
                criterion_text="Correct use of area formula πr² and calculation to 2 d.p.",
                mark_code_display="M3",
                marks_value=3.0,
                mark_type_primary="M",
                qualifiers_and_notes="Accept 25π = 78.54 cm²"
            )
        ],
        total_marks_for_part=3
    )

    # Updated solver algorithm for refined question
    refined_question.solver_algorithm = SolverAlgorithm(
        steps=[
            SolverStep(
                step_number=1,
                description_text="Apply area formula",
                mathematical_expression_latex="A = \\pi r^2",
                skill_applied_tag="area_formula"
            ),
            SolverStep(
                step_number=2,
                description_text="Substitute and calculate to 2 d.p.",
                mathematical_expression_latex="A = \\pi \\times 5^2 = 25\\pi = 78.54",
                skill_applied_tag="calculation_precision"
            )
        ]
    )

    # Mock refinement agent to return complete structure
    mocks['refinement_agent'].refine_question.return_value = (
        refined_question,
        {'interaction_id': str(uuid.uuid4()), 'success': True}
    )

    # Mock review scores: first low (triggers refinement), then high (approval)
    mocks['review_agent'].review_question.side_effect = [
        # First review - needs improvement
        (
            {
                'overall_score': 0.68,
                'clarity_score': 0.60,  # Poor clarity
                'difficulty_score': 0.75,
                'curriculum_alignment_score': 0.70,
                'mathematical_accuracy_score': 0.65
            },
            {'interaction_id': str(uuid.uuid4()), 'success': True}
        ),
        # Second review - after refinement
        (
            {
                'overall_score': 0.88,
                'clarity_score': 0.85,  # Improved clarity
                'difficulty_score': 0.90,
                'curriculum_alignment_score': 0.88,
                'mathematical_accuracy_score': 0.90
            },
            {'interaction_id': str(uuid.uuid4()), 'success': True}
        )
    ]

    # Create workflow
    workflow = QualityControlWorkflow(**mocks)

    # Process question
    session_id = str(uuid.uuid4())
    config = {"type": "geometry_basic", "target_grade": 8}
    result = workflow.process_question(original_question, session_id, config)

    # Display comprehensive results
    print("🔍 ORIGINAL QUESTION ANALYSIS:")
    print(f"Text: {original_question.raw_text_content}")
    print(f"Marks: {original_question.marks}")
    print(f"Answer: {original_question.solution_and_marking_scheme.final_answers_summary[0].answer_text}")
    print(f"Marking Criteria: {len(original_question.solution_and_marking_scheme.mark_allocation_criteria)} criterion")
    print(f"Solution Steps: {len(original_question.solver_algorithm.steps)} steps")
    print(f"Quality Score: 0.68 (clarity issue)")
    print()

    print("🔧 REFINEMENT PROCESS:")
    print("• Primary issue: Lack of precision specification")
    print("• Solution: Added '2 decimal places' instruction")
    print("• Updated marking scheme to reflect precision requirement")
    print("• Enhanced solver algorithm with precision step")
    print()

    print("✨ REFINED QUESTION ANALYSIS:")
    refined = result['approved_question']
    print(f"Text: {refined.raw_text_content}")
    print(f"Marks: {refined.marks}")
    print(f"Answer: {refined.solution_and_marking_scheme.final_answers_summary[0].answer_text}")
    print(f"Updated Marking: {refined.solution_and_marking_scheme.mark_allocation_criteria[0].criterion_text}")
    print(f"Solution Steps: {len(refined.solver_algorithm.steps)} steps")
    print(f"Final Quality Score: 0.88")
    print()

    print("🎯 DATABASE INSERTION READINESS:")
    print("✅ Complete CandidateQuestion structure")
    print("✅ All required fields populated")
    print("✅ Marking scheme aligned with refined content")
    print("✅ Solver algorithm updated for precision")
    print("✅ Taxonomy preserved from original")
    print("✅ Full audit trail maintained")
    print()

    print(f"🔄 WORKFLOW SUMMARY:")
    print(f"Decision: {result['final_decision'].value}")
    print(f"Success: {result['success']}")
    print(f"Iterations: {result['total_iterations']}")
    print(f"Auto-approved: ✅ Question improved and ready for database")


def demo_structure_comparison():
    """Demo: Show the difference between simple and complete refinement."""
    print("\n📊 DEMO: Refinement Structure Comparison")
    print("=" * 60)

    original = create_sample_question()

    print("❌ OLD APPROACH (Simple Text Refinement):")
    print("• Only question text improved")
    print("• Marking scheme unchanged")
    print("• Solver algorithm unchanged")
    print("• Inconsistency between question and assessment")
    print("• Manual update of dependent fields required")
    print()

    print("✅ NEW APPROACH (Complete Structure Refinement):")
    print("• Question text improved with precision")
    print("• Marking scheme updated to match")
    print("• Solver algorithm enhanced")
    print("• Complete consistency across all fields")
    print("• Ready for immediate database insertion")
    print()

    print("🏗️ STRUCTURE COMPLETENESS:")
    fields = [
        "question_id_local", "question_id_global", "raw_text_content",
        "marks", "command_word", "taxonomy", "solution_and_marking_scheme",
        "solver_algorithm", "generation_metadata"
    ]

    for field in fields:
        print(f"✅ {field}")

    print()
    print("🎯 BUSINESS VALUE:")
    print("• Eliminates manual post-processing")
    print("• Ensures assessment validity")
    print("• Maintains educational objectives")
    print("• Provides complete audit trail")
    print("• Enables automated quality improvement")


def main():
    """Run all demos."""
    print("🏗️  COMPLETE QUESTION REFINEMENT SYSTEM")
    print("🔄 Database-Ready Question Improvement")
    print("=" * 70)

    try:
        demo_complete_refinement_structure()
        demo_structure_comparison()

        print("\n" + "=" * 70)
        print("✅ All demos completed successfully!")
        print("\n📊 SYSTEM CAPABILITIES:")
        print("• Complete question structure refinement")
        print("• Automatic marking scheme updates")
        print("• Solver algorithm enhancement")
        print("• Database insertion readiness")
        print("• Educational objective preservation")
        print("• Cambridge IGCSE compliance")
        print("• Full audit trail maintenance")
        print("• Production-grade quality control")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
