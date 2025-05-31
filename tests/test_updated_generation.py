#!/usr/bin/env python3
"""Test question generation with updated validation and skill tags"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.generation_service import QuestionGenerationService
from src.validation.question_validator import validate_question

load_dotenv()

async def test_updated_generation():
    """Test question generation with enum validation"""
    print('🧪 Testing question generation with updated validation...')

    # Setup
    connection_string = os.getenv('NEON_DATABASE_URL')

    service = QuestionGenerationService(
        database_url=connection_string,
        debug=True
    )

    await service.initialize()

    # Generate one question
    from src.models import GenerationRequest, CalculatorPolicy

    request = GenerationRequest(
        target_grades=[5],
        count_per_grade=1,
        calculator_policy=CalculatorPolicy.NOT_ALLOWED
    )

    response = await service.generate_questions(request)

    print(f'✅ Generated {len(response.generated_questions)} questions')

    if response.generated_questions:
        q = response.generated_questions[0]
        print(f'\n📋 Question Details:')
        print(f'ID: {q.question_id_global}')
        print(f'Grade: {q.target_grade_input}')
        print(f'Marks: {q.marks}')
        print(f'Command: {q.command_word.value}')
        print(f'Content: {q.raw_text_content[:100]}...')

        print(f'\n🏷️ Taxonomy:')
        print(f'Skill tags: {q.taxonomy.skill_tags}')
        print(f'Subject refs: {q.taxonomy.subject_content_references}')
        print(f'Topic path: {q.taxonomy.topic_path}')

        # Test validation
        print(f'\n🔍 Running validation...')
        result = validate_question(q, verbose=True)

        if result.is_valid:
            print(f'\n✅ Question passed validation!')
        else:
            print(f'\n❌ Question failed validation')
            print(f'Critical errors: {result.critical_errors_count}')
            print(f'Warnings: {result.warnings_count}')

    await service.shutdown()

if __name__ == "__main__":
    asyncio.run(test_updated_generation())
