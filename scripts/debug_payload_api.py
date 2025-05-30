#!/usr/bin/env python3
"""
Debug Payload API - Test connectivity and permissions with proper authentication.

This script helps diagnose why payload tests are being skipped.
"""

import os
import sys
import json
import requests
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.services.payload_publisher import PayloadPublisher


def login_and_get_token():
    """Login to Payload and get a JWT token."""
    api_url = os.getenv('PAYLOAD_API_URL')
    email = os.getenv('PAYLOAD_EMAIL')
    password = os.getenv('PAYLOAD_PASSWORD')

    if not email or not password:
        print("❌ PAYLOAD_EMAIL and PAYLOAD_PASSWORD must be set in .env file")
        return None

    try:
        login_response = requests.post(
            f"{api_url}/users/login",
            json={
                "email": email,
                "password": password
            },
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if login_response.status_code == 200:
            data = login_response.json()
            token = data.get("token")
            print(f"✅ Login successful!")
            return token
        else:
            print(f"❌ Login failed: {login_response.status_code}")
            print(f"   Response: {login_response.text[:200]}...")
            return None

    except Exception as e:
        print(f"❌ Login error: {str(e)}")
        return None


async def test_payload_connectivity():
    """Test Payload API connectivity and permissions."""

    print("🔍 Payload API Diagnostics")
    print("=" * 50)

    # Check environment variables
    api_url = os.getenv('PAYLOAD_API_URL')
    email = os.getenv('PAYLOAD_EMAIL')
    password = os.getenv('PAYLOAD_PASSWORD')

    print(f"📡 API URL: {api_url}")
    print(f"🔑 Email: {'Set' if email else 'Not set'}")
    print(f"🔑 Password: {'Set' if password else 'Not set'}")

    if not email or not password:
        print("❌ PAYLOAD_EMAIL and PAYLOAD_PASSWORD not set!")
        print("   Add these to your .env file:")
        print("   PAYLOAD_EMAIL=your-admin-email@example.com")
        print("   PAYLOAD_PASSWORD=your-password")
        return False

    if not api_url:
        print("❌ PAYLOAD_API_URL not set!")
        return False

    # Test login and get JWT token
    print(f"\n🔐 Testing login...")
    token = login_and_get_token()
    if not token:
        return False

    # Test PayloadPublisher initialization
    print(f"\n🧪 Testing PayloadPublisher...")
    publisher = PayloadPublisher()
    print(f"   is_enabled(): {publisher.is_enabled()}")
    print(f"   API URL: {publisher.api_url}")
    print(f"   Has static token: {bool(publisher.api_token)}")

    # Test basic API connectivity with JWT token
    print(f"\n🌐 Testing API connectivity with JWT...")
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    try:
        # Test basic connection to root API
        response = requests.get(f"{api_url}", headers=headers, timeout=10)
        print(f"   Root API: {response.status_code}")
        if response.status_code != 200:
            print(f"   Response: {response.text[:200]}...")

    except Exception as e:
        print(f"   ❌ Connection failed: {str(e)}")
        return False

    # Test specific endpoints
    endpoints_to_test = [
        '/questions',
        '/solutionMarkingSchemes',
        '/solverAlgorithms'
    ]

    print(f"\n📋 Testing specific endpoints...")
    for endpoint in endpoints_to_test:
        try:
            response = requests.get(f"{api_url}{endpoint}", headers=headers, timeout=10)
            print(f"   {endpoint}: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                if 'docs' in data:
                    print(f"      Found {len(data['docs'])} items")
                elif isinstance(data, list):
                    print(f"      Found {len(data)} items")
                else:
                    print(f"      Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            else:
                print(f"      Error: {response.text[:100]}...")
        except Exception as e:
            print(f"   ❌ {endpoint}: {str(e)}")

    # Test creating a simple question
    print(f"\n📝 Testing question creation...")
    test_question_data = {
        'question_id_global': 'test_debug_001',
        'question_number_display': 'Debug Test',
        'marks': 1,
        'command_word': 'Calculate',
        'raw_text_content': 'Debug test question',
        'taxonomy': {
            'topic_path': ['Test'],
            'subject_content_references': ['C1.1'],
            'skill_tags': ['TEST_SKILL'],
            'cognitive_level': 'ProceduralFluency'
        },
        'source': 'debug_test'
    }

    try:
        response = requests.post(
            f"{api_url}/questions",
            json=test_question_data,
            headers=headers,
            timeout=10
        )
        print(f"   Create question: {response.status_code}")
        if response.status_code == 201:
            question_data = response.json()
            question_id = question_data.get('id')
            print(f"   ✅ Created question with ID: {question_id}")

            # Clean up - delete the test question
            if question_id:
                delete_response = requests.delete(
                    f"{api_url}/questions/{question_id}",
                    headers=headers,
                    timeout=10
                )
                print(f"   Cleanup: {delete_response.status_code}")
        else:
            print(f"   ❌ Failed: {response.text[:200]}...")

    except Exception as e:
        print(f"   ❌ Create test failed: {str(e)}")

    # Store the token for PayloadPublisher testing
    return token


async def test_sample_question_publish_with_jwt(jwt_token):
    """Test publishing a sample question using JWT token."""

    print(f"\n🧪 Testing Sample Question Publishing with JWT")
    print("=" * 50)

    from src.models.question_models import (
        CandidateQuestion, CommandWord, QuestionTaxonomy,
        SolutionAndMarkingScheme, SolverAlgorithm, AnswerSummary,
        MarkAllocationCriterion, SolverStep, LLMModel, GenerationStatus
    )
    import uuid
    from datetime import datetime

    # Create a minimal test question
    unique_id = str(uuid.uuid4())[:8]

    test_question = CandidateQuestion(
        question_id_local=f"debug_q_{unique_id}",
        question_id_global=f"debug_global_{unique_id}",
        question_number_display="Debug Question",
        marks=1,
        command_word=CommandWord.CALCULATE,
        raw_text_content=f"Debug test question {unique_id}",
        formatted_text_latex=None,
        taxonomy=QuestionTaxonomy(
            topic_path=["Debug", "Test"],
            subject_content_references=["C1.1"],
            skill_tags=["DEBUG_TEST"],
            cognitive_level="ProceduralFluency"
        ),
        solution_and_marking_scheme=SolutionAndMarkingScheme(
            final_answers_summary=[
                AnswerSummary(answer_text="Debug answer")
            ],
            mark_allocation_criteria=[
                MarkAllocationCriterion(
                    criterion_id="debug_1",
                    criterion_text="Debug criterion",
                    mark_code_display="B1",
                    marks_value=1.0,
                    mark_type_primary="B"
                )
            ],
            total_marks_for_part=1
        ),
        solver_algorithm=SolverAlgorithm(
            steps=[
                SolverStep(
                    step_number=1,
                    description_text="Debug step",
                    mathematical_expression_latex="x = 1"
                )
            ]
        ),
        generation_id=uuid.uuid4(),
        target_grade_input=5,
        llm_model_used_generation=LLMModel.GPT_4O.value,
        llm_model_used_marking_scheme=LLMModel.GPT_4O.value,
        prompt_template_version_generation="v1.0",
        prompt_template_version_marking_scheme="v1.0",
        generation_timestamp=datetime.utcnow(),
        status=GenerationStatus.CANDIDATE
    )

    # Test using direct API calls with JWT token
    api_url = os.getenv('PAYLOAD_API_URL')
    headers = {
        'Authorization': f'Bearer {jwt_token}',
        'Content-Type': 'application/json'
    }

    try:
        print(f"📝 Testing direct API calls with JWT...")

        # Test creating marking scheme
        marking_scheme_data = {
            'final_answers_summary': [
                {
                    'answer_text': answer.answer_text,
                    'value_numeric': answer.value_numeric,
                    'unit': answer.unit
                }
                for answer in test_question.solution_and_marking_scheme.final_answers_summary
            ],
            'mark_allocation_criteria': [
                {
                    'criterion_id': criterion.criterion_id,
                    'criterion_text': criterion.criterion_text,
                    'mark_code_display': criterion.mark_code_display,
                    'marks_value': criterion.marks_value,
                    'mark_type_primary': criterion.mark_type_primary,
                    'qualifiers_and_notes': getattr(criterion, 'qualifiers_and_notes', None)
                }
                for criterion in test_question.solution_and_marking_scheme.mark_allocation_criteria
            ],
            'total_marks_for_part': test_question.solution_and_marking_scheme.total_marks_for_part,
            'source': 'debug_test'
        }

        scheme_response = requests.post(
            f"{api_url}/solutionMarkingSchemes",
            json=marking_scheme_data,
            headers=headers,
            timeout=10
        )
        print(f"   Marking scheme: {scheme_response.status_code}")
        if scheme_response.status_code != 201:
            print(f"      Error: {scheme_response.text[:200]}...")
            return

        scheme_id = scheme_response.json().get('id')
        print(f"   ✅ Created marking scheme: {scheme_id}")

        # Test creating solver algorithm
        solver_data = {
            'steps': [
                {
                    'step_number': step.step_number,
                    'description_text': step.description_text,
                    'mathematical_expression_latex': step.mathematical_expression_latex,
                    'skill_applied_tag': getattr(step, 'skill_applied_tag', None),
                    'justification_or_reasoning': getattr(step, 'justification_or_reasoning', None)
                }
                for step in test_question.solver_algorithm.steps
            ],
            'source': 'debug_test'
        }

        solver_response = requests.post(
            f"{api_url}/solverAlgorithms",
            json=solver_data,
            headers=headers,
            timeout=10
        )
        print(f"   Solver algorithm: {solver_response.status_code}")
        if solver_response.status_code != 201:
            print(f"      Error: {solver_response.text[:200]}...")
            return

        solver_id = solver_response.json().get('id')
        print(f"   ✅ Created solver algorithm: {solver_id}")

        # Test creating main question
        question_data = {
            'question_id_global': test_question.question_id_global,
            'question_number_display': test_question.question_number_display,
            'marks': test_question.marks,
            'command_word': test_question.command_word.value,
            'raw_text_content': test_question.raw_text_content,
            'formatted_text_latex': test_question.formatted_text_latex,
            'taxonomy': {
                'topic_path': test_question.taxonomy.topic_path,
                'subject_content_references': test_question.taxonomy.subject_content_references,
                'skill_tags': test_question.taxonomy.skill_tags,
                'cognitive_level': test_question.taxonomy.cognitive_level,
                'difficulty_estimate_0_to_1': getattr(test_question.taxonomy, 'difficulty_estimate_0_to_1', None)
            },
            'solution_and_marking_scheme': scheme_id,
            'solver_algorithm': solver_id,
            'source': 'debug_test',
            'generation_metadata': {
                'original_question_id': test_question.question_id_local,
                'generation_id': str(test_question.generation_id),
                'target_grade': test_question.target_grade_input,
                'llm_model_used': test_question.llm_model_used_generation
            }
        }

        question_response = requests.post(
            f"{api_url}/questions",
            json=question_data,
            headers=headers,
            timeout=10
        )
        print(f"   Question creation: {question_response.status_code}")
        if question_response.status_code == 201:
            question_id = question_response.json().get('id')
            print(f"   ✅ Created question: {question_id}")

            # Test verification
            verify_response = requests.get(
                f"{api_url}/questions",
                params={'where[question_id_global][equals]': test_question.question_id_global},
                headers=headers,
                timeout=10
            )
            print(f"   Verification: {verify_response.status_code}")
            if verify_response.status_code == 200:
                found_questions = verify_response.json().get('docs', [])
                print(f"   ✅ Found {len(found_questions)} questions")

            # Cleanup
            for resource_id, endpoint in [
                (question_id, 'questions'),
                (solver_id, 'solverAlgorithms'),
                (scheme_id, 'solutionMarkingSchemes')
            ]:
                if resource_id:
                    delete_response = requests.delete(
                        f"{api_url}/{endpoint}/{resource_id}",
                        headers=headers,
                        timeout=10
                    )
                    print(f"   Cleanup {endpoint}: {delete_response.status_code}")
        else:
            print(f"   ❌ Failed: {question_response.text[:200]}...")

    except Exception as e:
        print(f"   ❌ Publishing exception: {str(e)}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all diagnostic tests."""

    print("🚀 Starting Payload API Diagnostics with JWT Authentication\n")

    # Test basic connectivity and get JWT token
    jwt_token = await test_payload_connectivity()

    if jwt_token:
        # Test actual question publishing with JWT
        await test_sample_question_publish_with_jwt(jwt_token)

    print(f"\n✅ Diagnostics complete!")
    print(f"\n💡 To fix the payload tests:")
    print(f"   1. Add PAYLOAD_EMAIL and PAYLOAD_PASSWORD to .env")
    print(f"   2. Update PayloadPublisher to use JWT authentication")


if __name__ == "__main__":
    asyncio.run(main())
