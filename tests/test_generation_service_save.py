#!/usr/bin/env python3
"""Quick test of generation service database saving"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.generation_service import QuestionGenerationService
from src.models import GenerationRequest, CalculatorPolicy

load_dotenv()

async def test_save():
    service = QuestionGenerationService(os.getenv('NEON_DATABASE_URL'), debug=True)
    await service.initialize()

    request = GenerationRequest(
        target_grades=[5],
        count_per_grade=1,
        calculator_policy=CalculatorPolicy.NOT_ALLOWED
    )

    print('🚀 Testing generation service database save...')
    response = await service.generate_questions(request)

    print(f'✅ Generated: {response.total_generated} questions')
    print(f'⏱️  Time: {response.generation_time_seconds:.2f}s')

    if response.generated_questions:
        q = response.generated_questions[0]
        print(f'📝 Question: {q.raw_text_content[:80]}...')
        print(f'🆔 ID: {q.question_id_global}')

    await service.shutdown()

if __name__ == "__main__":
    asyncio.run(test_save())
