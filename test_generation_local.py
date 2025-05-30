#!/usr/bin/env python3
"""
Test script for question generation using local file data
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from services.generation_service import QuestionGenerationService
from models.generation_config import GenerationConfig

async def test_generation_with_local_seed():
    """Test question generation using a local file seed question"""
    print("🧪 Testing question generation with local seed...")

    try:
        # Initialize service with debug enabled
        service = QuestionGenerationService(debug=True)

        # Test 1: Generate with a simple single-part question
        print("\n🎯 Test 1: Generate with Q1a (simple question) as seed")
        config = GenerationConfig(
            model="gpt-4o-mini",
            provider="openai",
            target_grade=6,
            seed_question_id="Q1a"  # Will use local file method
        )

        print(f"Config: {config}")

        # This should trigger the local file fallback since no database is connected
        result = await service.generate_question(config)

        if result:
            print("✅ Generation successful!")
            print(f"Question: {result.raw_text_content[:100]}...")
            print(f"Marks: {result.marks}")
            print(f"Command: {result.command_word.value}")
        else:
            print("❌ Generation failed")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def test_browse_then_generate():
    """Test browsing questions then generating with a selected seed"""
    print("\n🔍 Testing browse then generate workflow...")

    try:
        service = QuestionGenerationService(debug=True)

        # Get some questions without assets
        questions = await service.db_client.get_questions_from_local_file(
            limit=3,
            exclude_with_assets=True
        )

        if not questions:
            print("❌ No questions found")
            return

        print(f"✅ Found {len(questions)} questions without assets:")
        for i, q in enumerate(questions, 1):
            print(f"{i}. {q.get('question_id_local', 'N/A')} - {q.get('command_word', 'N/A')} ({q.get('marks', 0)} marks)")
            print(f"   {q.get('raw_text_content', '')[:80]}...")

        # Use the first question as seed
        seed_question = questions[0]
        seed_id = seed_question.get('question_id_local', 'Q1a')

        print(f"\n🎯 Generating new question inspired by: {seed_id}")

        config = GenerationConfig(
            model="gpt-4o-mini",
            provider="openai",
            target_grade=7,  # Slightly higher grade
            seed_question_id=seed_id
        )

        result = await service.generate_question(config)

        if result:
            print("✅ Generation with seed successful!")
            print(f"\nOriginal seed: {seed_question.get('raw_text_content', '')[:60]}...")
            print(f"Generated: {result.raw_text_content[:60]}...")
            print(f"Same topic: {result.taxonomy.topic_path}")
        else:
            print("❌ Generation with seed failed")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def test_question_set_generation():
    """Test generation using a multi-part question set as context"""
    print("\n🔢 Testing generation with multi-part question set...")

    try:
        service = QuestionGenerationService(debug=True)

        # Use Q11 which has multiple parts (a, b, c)
        seed_id = "Q11a"

        print(f"🎯 Getting question set for: {seed_id}")
        question_set = await service.db_client.get_question_set_from_local_file(seed_id)

        if question_set:
            print(f"✅ Found {len(question_set)} parts in question set:")
            for q in question_set:
                print(f"   - {q.get('question_id_local', 'N/A')}: {q.get('marks', 0)} marks - {q.get('command_word', 'N/A')}")
                print(f"     {q.get('raw_text_content', '')[:50]}...")

        print(f"\n🎯 Generating new multi-part question inspired by Q11 set...")

        config = GenerationConfig(
            model="gpt-4o-mini",
            provider="openai",
            target_grade=6,
            seed_question_id=seed_id  # This will trigger full question set context
        )

        result = await service.generate_question(config)

        if result:
            print("✅ Multi-part inspired generation successful!")
            print(f"Generated question: {result.raw_text_content[:100]}...")
            print(f"Marks: {result.marks}")
            print(f"Topic: {result.taxonomy.topic_path}")
        else:
            print("❌ Multi-part inspired generation failed")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting question generation tests with local file data...")
    asyncio.run(test_generation_with_local_seed())
    asyncio.run(test_browse_then_generate())
    asyncio.run(test_question_set_generation())
