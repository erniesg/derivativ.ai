#!/usr/bin/env python3
"""
Simple test script to validate local question loading functionality
"""

import asyncio
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from database.neon_client import NeonDBClient

async def test_local_questions():
    """Test loading questions from local file"""
    print("🧪 Testing local question loading...")

    # Initialize client (won't actually connect to DB for local operations)
    client = NeonDBClient(database_url="dummy://for/local/testing")

    print("\n📖 Loading questions from local file...")
    try:
        questions = await client.get_questions_from_local_file(
            limit=5,
            exclude_with_assets=True
        )

        if not questions:
            print("❌ No questions found in local file")
            return

        print(f"✅ Found {len(questions)} questions without assets:")
        print("=" * 80)

        for i, q in enumerate(questions, 1):
            taxonomy = q.get('taxonomy', {})
            topic_path = taxonomy.get('topic_path', [])
            if isinstance(topic_path, list):
                topic_str = ' > '.join(topic_path)
            else:
                topic_str = str(topic_path)

            print(f"{i}. {q.get('question_id_global', 'N/A')}")
            print(f"   Command: {q.get('command_word', 'N/A')} | Marks: {q.get('marks', 0)}")
            print(f"   Topic: {topic_str}")
            print(f"   Text: {q.get('raw_text_content', '')[:80]}...")
            print()

        # Test getting a full question set
        if questions:
            # Try Q10a which should have multiple parts (a, b, c)
            test_id = "Q10a"
            print(f"\n🔍 Testing full question set for: {test_id}")

            question_set = await client.get_question_set_from_local_file(test_id)

            if question_set:
                print(f"✅ Found {len(question_set)} parts in question set:")
                for q in question_set:
                    print(f"   - {q.get('question_id_local', 'N/A')}: {q.get('marks', 0)} marks - {q.get('command_word', 'N/A')}")
            else:
                print("❌ No question set found")

            # Also test the first question from our results
            first_id = questions[0].get('question_id_local', 'Q1a')
            print(f"\n🔍 Testing full question set for: {first_id}")

            question_set2 = await client.get_question_set_from_local_file(first_id)

            if question_set2:
                print(f"✅ Found {len(question_set2)} parts in question set:")
                for q in question_set2:
                    print(f"   - {q.get('question_id_local', 'N/A')}: {q.get('marks', 0)} marks - {q.get('command_word', 'N/A')}")
            else:
                print("❌ No question set found")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def test_file_structure():
    """Test if the local file exists and is readable"""
    print("📁 Checking file structure...")

    file_path = "data/processed/2025p1.json"
    if os.path.exists(file_path):
        print(f"✅ Found {file_path}")

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            questions = data.get('questions', [])
            print(f"✅ File contains {len(questions)} questions")

            # Check for questions without assets
            no_assets = [q for q in questions if not q.get('assets', [])]
            print(f"✅ {len(no_assets)} questions without assets")

            # Show sample question structure
            if questions:
                sample = questions[0]
                print(f"\n📋 Sample question structure:")
                print(f"   - question_id_global: {sample.get('question_id_global', 'N/A')}")
                print(f"   - question_id_local: {sample.get('question_id_local', 'N/A')}")
                print(f"   - marks: {sample.get('marks', 'N/A')}")
                print(f"   - command_word: {sample.get('command_word', 'N/A')}")
                print(f"   - has_assets: {bool(sample.get('assets', []))}")

        except Exception as e:
            print(f"❌ Error reading file: {e}")
    else:
        print(f"❌ File not found: {file_path}")

async def explore_question_patterns():
    """Explore the actual question ID patterns in the file"""
    print("🔍 Exploring question ID patterns...")

    # Use the working JSON loading method from the database client
    client = NeonDBClient(database_url="dummy://for/local/testing")

    try:
        file_path = "data/processed/2025p1.json"
        paper_data = client._safe_json_load(file_path)

        questions = paper_data.get('questions', [])
        print(f"✅ Loaded {len(questions)} questions successfully")

        # Group questions by base number
        question_groups = {}
        standalone_questions = []

        for q in questions:
            local_id = q.get('question_id_local', '')
            global_id = q.get('question_id_global', '')

            # Extract base number from local_id (e.g., Q1a -> 1, Q10b -> 10, Q15aii -> 15)
            if local_id.startswith('Q') and len(local_id) > 1:
                # Find where the letter starts
                base_num = ''
                for i, char in enumerate(local_id[1:], 1):
                    if char.isdigit():
                        base_num += char
                    else:
                        break

                if base_num:
                    if base_num not in question_groups:
                        question_groups[base_num] = []
                    question_groups[base_num].append({
                        'local_id': local_id,
                        'global_id': global_id,
                        'marks': q.get('marks', 0),
                        'command': q.get('command_word', 'N/A'),
                        'has_assets': bool(q.get('assets', []))
                    })
                else:
                    standalone_questions.append(local_id)
            else:
                standalone_questions.append(local_id)

        # Show multi-part questions
        multi_part = {k: v for k, v in question_groups.items() if len(v) > 1}
        single_part = {k: v for k, v in question_groups.items() if len(v) == 1}

        print(f"\n📊 Question Analysis:")
        print(f"   - Multi-part questions: {len(multi_part)}")
        print(f"   - Single-part questions: {len(single_part)}")
        print(f"   - Standalone/other: {len(standalone_questions)}")

        print(f"\n🔢 Multi-part questions (first 10):")
        for i, (base_num, parts) in enumerate(sorted(multi_part.items(), key=lambda x: int(x[0]))):
            if i >= 10:  # Show only first 10
                print(f"   ... and {len(multi_part) - 10} more")
                break
            print(f"   Q{base_num}: {len(parts)} parts")
            for part in sorted(parts, key=lambda x: x['local_id']):
                asset_indicator = "🖼️" if part['has_assets'] else "  "
                print(f"     {asset_indicator} {part['local_id']:8s} | {part['command']:12s} | {part['marks']} marks")

        # Test our question set function with actual multi-part questions
        if multi_part:
            # Test with Q15 since we saw Q15aii and Q15bii in the results
            test_cases = ["Q15a", "Q15aii", "Q2a", "Q21a"]

            for test_id in test_cases:
                if any(test_id[1:-1] in multi_part.keys() for test_id in [test_id]):  # Check if base number exists
                    print(f"\n🧪 Testing question set retrieval for: {test_id}")
                    question_set = await client.get_question_set_from_local_file(test_id)

                    if question_set:
                        print(f"✅ Retrieved {len(question_set)} parts:")
                        for q in question_set:
                            print(f"   - {q.get('question_id_local', 'N/A')}: {q.get('marks', 0)} marks - {q.get('command_word', 'N/A')}")
                    else:
                        print(f"❌ Failed to retrieve question set for {test_id}")

                        # Debug the pattern matching
                        print(f"🔧 Debugging pattern matching...")

                        # Extract base number properly
                        base_num = ''
                        for char in test_id[1:]:
                            if char.isdigit():
                                base_num += char
                            else:
                                break

                        print(f"   Test ID: {test_id}")
                        print(f"   Extracted base number: '{base_num}'")
                        print(f"   Looking for pattern: Q{base_num}*")

                        matches = []
                        for q in questions:
                            local_id = q.get("question_id_local", "")
                            if local_id.startswith(f"Q{base_num}"):
                                matches.append(local_id)

                        print(f"   Found matches: {matches}")
                    break  # Test only the first valid case

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_file_structure())
    asyncio.run(explore_question_patterns())
    asyncio.run(test_local_questions())
