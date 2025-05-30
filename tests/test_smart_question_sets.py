#!/usr/bin/env python3
"""
Test script for the new intelligent question set detection
"""

import asyncio
import sys
import os

print("🔄 Starting test script...")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("🔄 Importing modules...")

try:
    from database.neon_client import NeonDBClient
    print("✅ Import successful!")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

async def test_intelligent_question_sets():
    """Test the new intelligent question set detection"""
    print("🧠 Testing intelligent question set detection...")

    # Initialize client (won't actually connect to DB for local operations)
    client = NeonDBClient(database_url="dummy://for/local/testing")

    print("\n📊 First, let's analyze the question relationships...")

    try:
        # Load all questions and analyze relationships
        paper_data = client._safe_json_load("data/processed/2025p1.json")
        all_questions = paper_data.get('questions', [])

        print(f"✅ Loaded {len(all_questions)} questions")

        # Analyze relationships
        question_sets = client._analyze_question_relationships(all_questions)

        print(f"✅ Found {len(question_sets)} multi-part question sets:")

        for set_key, questions in sorted(question_sets.items()):
            print(f"\n🔢 {set_key}: {len(questions)} parts")
            for q in sorted(questions, key=lambda x: x.get('question_id_local', '')):
                asset_indicator = "🖼️" if q.get('assets') else "  "
                print(f"   {asset_indicator} {q.get('question_id_local', 'N/A'):8s} | {q.get('command_word', 'N/A'):12s} | {q.get('marks', 0):2d} marks")

        # Test intelligent retrieval for various question IDs
        test_cases = [
            "Q1a",           # Start of a 3-part set
            "Q11b",          # Middle of a 3-part set
            "Q15ai",         # Complex multi-part set (6 parts)
            "Q21a",          # Start of a 2-part set
            "0580_SP_25_P1_q10b",  # Global ID format
            "Q3",            # Single question (should not be in a set)
        ]

        print(f"\n🎯 Testing intelligent retrieval for various question IDs...")

        for test_id in test_cases:
            print(f"\n🔍 Testing: {test_id}")

            # Find the set key
            set_key = client._find_question_set_key(test_id, all_questions)

            if set_key:
                print(f"   ✅ Found in set: {set_key}")

                # Get the full set using intelligent method
                question_set = await client.get_intelligent_question_set(test_id, source="local")

                if question_set:
                    print(f"   ✅ Retrieved {len(question_set)} questions in set:")
                    for q in question_set:
                        print(f"      - {q.get('question_id_local', 'N/A')}: {q.get('marks', 0)} marks - {q.get('command_word', 'N/A')}")
                else:
                    print(f"   ❌ Failed to retrieve question set")
            else:
                print(f"   ℹ️  Not part of a multi-part set (standalone question)")

                # Should still return the single question
                question_set = await client.get_intelligent_question_set(test_id, source="local")
                if question_set and len(question_set) == 1:
                    q = question_set[0]
                    print(f"   ✅ Retrieved single question: {q.get('question_id_local', 'N/A')} - {q.get('command_word', 'N/A')}")
                else:
                    print(f"   ❌ Failed to retrieve question")

        # Test edge cases
        print(f"\n🔧 Testing edge cases...")

        edge_cases = [
            "NonExistent",      # Question that doesn't exist
            "Q99a",            # Question number that doesn't exist
            "",                # Empty string
        ]

        for test_id in edge_cases:
            print(f"\n🔍 Edge case: '{test_id}'")
            question_set = await client.get_intelligent_question_set(test_id, source="local")
            print(f"   Result: {len(question_set) if question_set else 0} questions")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Running intelligent question set tests...")
    asyncio.run(test_intelligent_question_sets())
    print("✅ Tests completed!")
