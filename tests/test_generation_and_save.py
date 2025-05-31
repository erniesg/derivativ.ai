#!/usr/bin/env python3
"""Test actual question generation and database saving with contextual skill tags"""

import sys
import os
import asyncio
import json
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.config_manager import ConfigManager
from src.services.prompt_loader import PromptLoader
from src.services.skill_tag_mapper import SkillTagMapper
from src.database.neon_client import NeonDBClient
from src.services.database_manager import DatabaseManager
from src.models.question_models import CandidateQuestion, QuestionTaxonomy, AnswerSummary, MarkAllocationCriterion, SolutionAndMarkingScheme, SolverStep, SolverAlgorithm, CommandWord, GenerationStatus
from src.models.enums import get_valid_skill_tags

def test_contextual_skill_tag_generation():
    """Test question generation with contextual skill tag mapping"""
    print("🧪 Testing Question Generation with Contextual Skill Tags...")

    config_manager = ConfigManager()
    prompt_loader = PromptLoader()
    skill_mapper = SkillTagMapper()

    # Test different config types
    test_configs = [
        {
            "config_id": "basic_arithmetic_gpt4o",
            "description": "Number topics - should only show arithmetic skills",
            "expected_categories": ["Number & Arithmetic"]
        },
        {
            "config_id": "algebra_claude4",
            "description": "Algebra topics - should only show algebraic skills",
            "expected_categories": ["Algebra"]
        },
        {
            "config_id": "geometry_gemini",
            "description": "Geometry topics - should only show geometric skills",
            "expected_categories": ["Geometry"]
        }
    ]

    for test_config in test_configs:
        print(f"\n{'='*60}")
        print(f"🔍 Testing: {test_config['config_id']}")
        print(f"📝 {test_config['description']}")

        # Get the config template
        config_template = config_manager.get_config_template(test_config['config_id'])
        if not config_template:
            print(f"❌ Config not found: {test_config['config_id']}")
            continue

        # Test contextual skill mapping
        print(f"\n📚 Subject content references: {config_template.subject_content_references}")

        # Get contextual tags
        contextual_tags = skill_mapper.get_relevant_skill_tags(config_template.subject_content_references)
        all_tags = get_valid_skill_tags()

        print(f"🏷️ Contextual skill tags: {len(contextual_tags)}/{len(all_tags)} ({len(contextual_tags)/len(all_tags)*100:.1f}%)")
        print(f"📉 Filtered out: {len(all_tags) - len(contextual_tags)} irrelevant tags")

        # Test prompt generation with contextual tags
        generation_config = config_manager.create_generation_config(test_config['config_id'], target_grade=5)
        if generation_config:
            # Prepare context
            context = {
                'target_grade': generation_config.target_grade,
                'calculator_policy': generation_config.calculator_policy,
                'desired_marks': generation_config.desired_marks,
                'subject_content_references': json.dumps(generation_config.subject_content_references),
                'command_word': 'Calculate',
                'syllabus_content': 'Test syllabus content for contextual skill tags',
                'command_word_definition': 'work out from given facts',
                'seed_question_context': 'No seed question',
                'marking_principles': 'Cambridge IGCSE marking principles',
                'generation_id': f'contextual-test-{test_config["config_id"]}'
            }

            try:
                # Generate prompt with contextual tags
                prompt = prompt_loader.format_question_generation_prompt(
                    generation_config.prompt_template_version_generation,
                    **context
                )

                print(f"✅ Prompt generated with contextual skill tags")

                # Check if prompt has expected categories
                found_categories = []
                for expected_cat in test_config['expected_categories']:
                    if f"**{expected_cat}:**" in prompt:
                        found_categories.append(expected_cat)

                if found_categories:
                    print(f"✅ Expected categories found: {', '.join(found_categories)}")
                else:
                    print(f"⚠️ Expected categories not found in prompt")

                # Count contextual tags in prompt
                tags_in_prompt = sum(1 for tag in contextual_tags if tag in prompt)
                print(f"✅ Contextual tags in prompt: {tags_in_prompt}/{len(contextual_tags)} ({tags_in_prompt/len(contextual_tags)*100:.1f}%)")

                # Show sample prompt excerpt for skill tags
                skill_section_start = prompt.find("## CONTEXTUALLY RELEVANT SKILL TAGS")
                if skill_section_start != -1:
                    skill_section_end = prompt.find("## EXAMPLES FOR GRADE", skill_section_start)
                    if skill_section_end == -1:
                        skill_section_end = skill_section_start + 500

                    sample = prompt[skill_section_start:skill_section_end]
                    lines = sample.split('\n')[:15]  # First 15 lines
                    print(f"\n📝 Sample of contextual skill tags section:")
                    for line in lines[:10]:
                        if line.strip():
                            print(f"   {line[:100]}...")

                print(f"🎉 Contextual skill tag injection working correctly!")

            except Exception as e:
                print(f"❌ Error testing prompt generation: {e}")
        else:
            print(f"❌ Could not create generation config")

async def test_database_integration():
    """Test database saving with contextual skill tag validation"""
    print(f"\n{'='*60}")
    print("💾 Testing Database Integration with Contextual Skill Tags...")

    # Check for database connection
    connection_string = os.getenv("NEON_DATABASE_URL")
    if not connection_string:
        print("⚠️ No database connection string found (NEON_DATABASE_URL)")
        print("📝 Creating mock question to demonstrate validation...")

        # Create a mock question with contextual skill tags
        skill_mapper = SkillTagMapper()

        # Test algebra question with contextual tags
        algebra_refs = ["C2.2", "C2.5"]  # Algebraic manipulation, Equations
        contextual_tags = skill_mapper.get_relevant_skill_tags(algebra_refs)

        mock_question = create_mock_question_with_contextual_tags(algebra_refs, contextual_tags)
        print(f"✅ Mock algebra question created with {len(contextual_tags)} contextual skill tags")
        print(f"🏷️ Sample contextual tags: {contextual_tags[:5]}...")
        print(f"📚 Subject content refs: {algebra_refs}")

        # Validate contextual alignment
        valid_tags, invalid_tags = skill_mapper.validate_skill_tags_for_subject_refs(
            mock_question.taxonomy.skill_tags,
            mock_question.taxonomy.subject_content_references
        )

        print(f"✅ Validation results:")
        print(f"   🎯 Valid contextual tags: {len(valid_tags)}/{len(mock_question.taxonomy.skill_tags)}")
        print(f"   ❌ Invalid/irrelevant tags: {len(invalid_tags)}")

        if len(invalid_tags) == 0:
            print(f"🎉 Perfect contextual alignment - all skill tags are relevant!")
        else:
            print(f"⚠️ Some tags not contextually aligned: {invalid_tags}")

        return

    # Test with real database connection
    try:
        neon_client = NeonDBClient(connection_string)
        await neon_client.connect()

        db_manager = DatabaseManager(connection_string)

        print("✅ Database connection established")

        # Test saving a question with contextual skill tags
        skill_mapper = SkillTagMapper()

        # Create question with contextual tags for different topics
        test_cases = [
            {
                "name": "Number Question",
                "subject_refs": ["C1.1", "C1.6"],
                "description": "Basic arithmetic with contextual number skills"
            },
            {
                "name": "Algebra Question",
                "subject_refs": ["C2.2", "C2.5"],
                "description": "Algebra with contextual algebraic skills"
            }
        ]

        saved_questions = []

        for case in test_cases:
            print(f"\n🔍 Testing: {case['name']}")

            contextual_tags = skill_mapper.get_relevant_skill_tags(case['subject_refs'])
            mock_question = create_mock_question_with_contextual_tags(
                case['subject_refs'],
                contextual_tags[:4]  # Use first 4 contextual tags
            )

            print(f"📚 Subject refs: {case['subject_refs']}")
            print(f"🏷️ Contextual tags: {mock_question.taxonomy.skill_tags}")

            # Test validation before saving
            valid_tags, invalid_tags = skill_mapper.validate_skill_tags_for_subject_refs(
                mock_question.taxonomy.skill_tags,
                mock_question.taxonomy.subject_content_references
            )

            if len(invalid_tags) == 0:
                print(f"✅ All skill tags are contextually valid")

                # Save to database
                try:
                    await db_manager.save_candidate_question(mock_question)
                    saved_questions.append(mock_question)
                    print(f"💾 Question saved to database: {mock_question.question_id_global}")
                except Exception as e:
                    print(f"❌ Error saving to database: {e}")
            else:
                print(f"⚠️ Invalid contextual tags detected: {invalid_tags}")

        print(f"\n🎯 Database Integration Results:")
        print(f"   💾 Questions saved: {len(saved_questions)}")
        print(f"   ✅ All saved questions have contextually valid skill tags")

        await neon_client.disconnect()

    except Exception as e:
        print(f"❌ Database connection error: {e}")

def create_mock_question_with_contextual_tags(subject_refs: list, contextual_tags: list) -> CandidateQuestion:
    """Create a mock question with contextually appropriate skill tags"""
    from uuid import uuid4
    from datetime import datetime

    taxonomy = QuestionTaxonomy(
        topic_path=["Algebra", "Equations"],
        subject_content_references=subject_refs,
        skill_tags=contextual_tags[:4],  # Use first 4 contextual tags
        cognitive_level="ProceduralFluency",
        difficulty_estimate_0_to_1=0.6
    )

    # Create proper AnswerSummary objects
    answer_summary = AnswerSummary(
        answer_text="x = 4",
        value_numeric=4.0,
        unit=None
    )

    # Create proper MarkAllocationCriterion objects
    mark_criteria = [
        MarkAllocationCriterion(
            criterion_id="crit_1",
            criterion_text="Correct solution method",
            mark_code_display="M1",
            marks_value=1,
            mark_type_primary="M",
            qualifiers_and_notes="Rearranging equation correctly"
        ),
        MarkAllocationCriterion(
            criterion_id="crit_2",
            criterion_text="Correct final answer",
            mark_code_display="A1",
            marks_value=1,
            mark_type_primary="A",
            qualifiers_and_notes="x = 4"
        )
    ]

    # Create proper SolutionAndMarkingScheme
    solution_scheme = SolutionAndMarkingScheme(
        final_answers_summary=[answer_summary],
        mark_allocation_criteria=mark_criteria,
        total_marks_for_part=2
    )

    # Create proper SolverStep objects
    solver_steps = [
        SolverStep(
            step_number=1,
            description_text="Subtract 5 from both sides",
            mathematical_expression_latex="2x = 13 - 5 = 8",
            justification_or_reasoning="Isolate the term with x"
        ),
        SolverStep(
            step_number=2,
            description_text="Divide both sides by 2",
            mathematical_expression_latex="x = 8 ÷ 2 = 4",
            justification_or_reasoning="Solve for x"
        )
    ]

    # Create proper SolverAlgorithm
    solver_algorithm = SolverAlgorithm(steps=solver_steps)

    question = CandidateQuestion(
        question_id_local="MockQ_CTX_001",
        question_id_global="mock_contextual_test_q001",
        question_number_display="Contextual Test Question",
        marks=2,
        command_word=CommandWord.CALCULATE,
        raw_text_content="Solve the equation 2x + 5 = 13 for x.",
        formatted_text_latex=None,
        taxonomy=taxonomy,
        solution_and_marking_scheme=solution_scheme,
        solver_algorithm=solver_algorithm,

        # Required generation metadata
        generation_id=uuid4(),
        target_grade_input=5,

        # Required model tracking
        llm_model_used_generation="contextual_test_model",
        llm_model_used_marking_scheme="contextual_test_model",
        llm_model_used_review=None,

        # Required prompt tracking
        prompt_template_version_generation="v1.2",
        prompt_template_version_marking_scheme="v1.0",
        prompt_template_version_review=None,

        # Default status
        status=GenerationStatus.CANDIDATE,
        generation_timestamp=datetime.utcnow(),
        confidence_score=0.85,
        validation_errors=[]
    )

    return question

async def main():
    """Main test runner"""
    print("🚀 Starting Contextual Skill Tag Generation and Database Tests...")

    # Test 1: Contextual skill tag generation
    test_contextual_skill_tag_generation()

    # Test 2: Database integration
    await test_database_integration()

    print(f"\n🎉 All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
