#!/usr/bin/env python3
"""
Step-by-step Payload API Testing - Test each collection individually.
"""

import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def get_jwt_token():
    """Get JWT token from Payload."""
    api_url = os.getenv('PAYLOAD_API_URL')
    email = os.getenv('PAYLOAD_EMAIL')
    password = os.getenv('PAYLOAD_PASSWORD')

    response = requests.post(
        f"{api_url}/users/login",
        json={"email": email, "password": password},
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        return response.json().get("token")
    else:
        print(f"Login failed: {response.text}")
        return None

def test_mark_criteria_creation():
    """Test creating a mark criterion - but now we need the scheme first."""
    print("\n🧪 Testing Mark Criteria Creation (with parent scheme)")
    print("=" * 40)

    token = get_jwt_token()
    if not token:
        return None, None

    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    # First create the parent scheme
    scheme_data = {
        'final_answers_summary': [
            {
                'answer_text': 'Test answer',
                'value_numeric': 42,
                'unit': 'cm'
            }
        ],
        'total_marks_for_part': 1
        # Note: mark_allocation_criteria will be added later
    }

    api_url = os.getenv('PAYLOAD_API_URL')
    response = requests.post(
        f"{api_url}/solutionMarkingSchemes",
        json=scheme_data,
        headers=headers
    )

    if response.status_code != 201:
        print(f"❌ Failed to create scheme: {response.text}")
        return None, token

    scheme = response.json()
    scheme_id = scheme.get('id')
    print(f"✅ Created parent scheme: {scheme_id}")

    # Then create the criterion with parent relationship
    criterion_data = {
        'criterion_id': 'test_crit_001',
        'criterion_text': 'Test criterion',
        'marks_value': 1.0,
        'solutionMarkingScheme': scheme_id  # Link to parent
    }

    response = requests.post(
        f"{api_url}/markCriteria",
        json=criterion_data,
        headers=headers
    )

    print(f"Criterion Status: {response.status_code}")
    if response.status_code == 201:
        data = response.json()
        criterion_id = data.get('id')
        print(f"✅ Created criterion: {criterion_id}")

        # Update scheme with criterion relationship
        update_data = {'mark_allocation_criteria': [criterion_id]}
        requests.patch(
            f"{api_url}/solutionMarkingSchemes/{scheme_id}",
            json=update_data,
            headers=headers
        )
        print(f"✅ Linked criterion to scheme")

        return (criterion_id, scheme_id), token
    else:
        print(f"❌ Failed: {response.text}")
        return None, token

def test_solution_marking_scheme_creation(criterion_data, token):
    """Test solution marking scheme - already created in previous step."""
    print("\n🧪 Testing Solution Marking Scheme (already created)")
    print("=" * 40)

    if not criterion_data:
        print("❌ No criterion data provided")
        return None

    criterion_id, scheme_id = criterion_data
    print(f"✅ Using existing scheme: {scheme_id}")
    return scheme_id

def test_solver_step_creation():
    """Test creating a solver step - but now we need the algorithm first."""
    print("\n🧪 Testing Solver Step Creation (with parent algorithm)")
    print("=" * 40)

    token = get_jwt_token()
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    # First create the parent algorithm
    algorithm_data = {}  # Empty container

    api_url = os.getenv('PAYLOAD_API_URL')
    response = requests.post(
        f"{api_url}/solverAlgorithms",
        json=algorithm_data,
        headers=headers
    )

    if response.status_code != 201:
        print(f"❌ Failed to create algorithm: {response.text}")
        return None, token

    algorithm = response.json()
    algorithm_id = algorithm.get('id')
    print(f"✅ Created parent algorithm: {algorithm_id}")

    # Then create the step with parent relationship
    step_data = {
        'step_number': 1,
        'description_text': 'Test step description',
        'solverAlgorithm': algorithm_id  # Link to parent
    }

    response = requests.post(
        f"{api_url}/solverSteps",
        json=step_data,
        headers=headers
    )

    print(f"Step Status: {response.status_code}")
    if response.status_code == 201:
        data = response.json()
        step_id = data.get('id')
        print(f"✅ Created step: {step_id}")

        # Update algorithm with step relationship
        update_data = {'steps': [step_id]}
        requests.patch(
            f"{api_url}/solverAlgorithms/{algorithm_id}",
            json=update_data,
            headers=headers
        )
        print(f"✅ Linked step to algorithm")

        return (step_id, algorithm_id), token
    else:
        print(f"❌ Failed: {response.text}")
        return None, token

def test_solver_algorithm_creation(step_data, token):
    """Test solver algorithm - already created in previous step."""
    print("\n🧪 Testing Solver Algorithm (already created)")
    print("=" * 40)

    if not step_data:
        print("❌ No step data provided")
        return None

    step_id, algorithm_id = step_data
    print(f"✅ Using existing algorithm: {algorithm_id}")
    return algorithm_id

def test_question_creation(scheme_id, algorithm_id):
    """Test creating a question."""
    print("\n🧪 Testing Question Creation")
    print("=" * 40)

    if not scheme_id or not algorithm_id:
        print("❌ Missing scheme or algorithm ID")
        return None

    token = get_jwt_token()
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    # Test with minimal data
    question_data = {
        'question_id_local': 'test_q_001',
        'question_id_global': 'test_global_001',
        'question_number_display': 'Test Question',
        'marks': 1,
        'command_word': 'Calculate',
        'raw_text_content': 'Test question content',
        'origin': 'llm_generated',  # Required field
        'taxonomy': {
            'topic_path': ['Test'],
            'subject_content_references': ['C1.1'],
            'skill_tags': ['TEST_SKILL'],
            'cognitive_level': 'ProceduralFluency'
        },
        'solution_and_marking_scheme': scheme_id,  # Relationship
        'solver_algorithm': algorithm_id,  # Relationship
        'assets': []  # Empty assets array
    }

    api_url = os.getenv('PAYLOAD_API_URL')
    response = requests.post(
        f"{api_url}/questions",
        json=question_data,
        headers=headers
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 201:
        data = response.json()
        question_id = data.get('id')
        print(f"✅ Created question: {question_id}")
        return question_id
    else:
        print(f"❌ Failed: {response.text}")
        return None

def cleanup_test_data(question_id, algorithm_id, scheme_id, step_id, criterion_id):
    """Clean up test data."""
    print("\n🧹 Cleaning up test data")
    print("=" * 40)

    token = get_jwt_token()
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    api_url = os.getenv('PAYLOAD_API_URL')

    # Delete in reverse order to handle relationships
    resources = [
        (question_id, 'questions'),
        (algorithm_id, 'solverAlgorithms'),
        (scheme_id, 'solutionMarkingSchemes'),
        (step_id, 'solverSteps'),
        (criterion_id, 'markCriteria')
    ]

    for resource_id, collection in resources:
        if resource_id:
            try:
                response = requests.delete(
                    f"{api_url}/{collection}/{resource_id}",
                    headers=headers
                )
                print(f"   {collection}: {response.status_code}")
            except Exception as e:
                print(f"   {collection}: Error - {str(e)}")

def main():
    """Run step-by-step tests."""
    print("🚀 Step-by-Step Payload API Testing")
    print("=" * 50)

    # Test 1: Create mark criterion
    criterion_data, token = test_mark_criteria_creation()

    # Test 2: Create solution marking scheme
    scheme_id = test_solution_marking_scheme_creation(criterion_data, token)

    # Test 3: Create solver step
    step_data, token = test_solver_step_creation()

    # Test 4: Create solver algorithm
    algorithm_id = test_solver_algorithm_creation(step_data, token)

    # Test 5: Create question
    question_id = test_question_creation(scheme_id, algorithm_id)

    if question_id:
        print(f"\n🎉 SUCCESS! Created complete question with ID: {question_id}")
    else:
        print(f"\n❌ FAILED to create complete question")

    # Cleanup
    step_id = step_data[0] if step_data else None
    criterion_id = criterion_data[0] if criterion_data else None
    cleanup_test_data(question_id, algorithm_id, scheme_id, step_id, criterion_id)

    print("\n✅ Test complete!")

if __name__ == "__main__":
    main()
