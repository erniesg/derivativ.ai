#!/usr/bin/env python3
"""Test the enhanced validation system with enum-based validation"""

import asyncio
import os
import json
import sys
from dotenv import load_dotenv

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validation.question_validator import validate_question, CambridgeQuestionValidator
from src.database.neon_client import NeonDBClient
from src.models.question_models import CandidateQuestion

load_dotenv()

async def test_validation():
    """Test validation on existing questions"""
    print("🔍 Testing Enhanced Validation System...")

    # Initialize validator
    validator = CambridgeQuestionValidator()

    print(f"✅ Loaded {len(validator.valid_skill_tags)} skill tags")
    print(f"✅ Loaded {len(validator.valid_subject_refs)} subject references")
    print(f"✅ Loaded {len(validator.valid_topic_paths)} topic paths")

    # Test with database questions
    connection_string = os.getenv('NEON_DATABASE_URL')
    if connection_string:
        client = NeonDBClient(connection_string)
        await client.connect()

        # Get recent questions
        questions = await client.get_candidate_questions(limit=3)

        if questions:
            print(f"\n📋 Testing {len(questions)} recent questions from database:")

            for i, question_data in enumerate(questions, 1):
                print(f"\n--- Question {i} ---")
                print(f"ID: {question_data.get('question_id', 'Unknown')}")

                # Check skill tags
                skill_tags = question_data.get('taxonomy', {}).get('skill_tags', [])
                print(f"Skill tags: {skill_tags}")

                valid_skills = [tag for tag in skill_tags if tag in validator.valid_skill_tags]
                invalid_skills = [tag for tag in skill_tags if tag not in validator.valid_skill_tags]

                if valid_skills:
                    print(f"✅ Valid skills: {valid_skills}")
                if invalid_skills:
                    print(f"❌ Invalid skills: {invalid_skills}")

                # Check subject refs
                subject_refs = question_data.get('taxonomy', {}).get('subject_content_references', [])
                print(f"Subject refs: {subject_refs}")

                valid_refs = [ref for ref in subject_refs if ref in validator.valid_subject_refs]
                invalid_refs = [ref for ref in subject_refs if ref not in validator.valid_subject_refs]

                if valid_refs:
                    print(f"✅ Valid refs: {valid_refs}")
                if invalid_refs:
                    print(f"❌ Invalid refs: {invalid_refs}")

                # Check topic paths
                topic_paths = question_data.get('taxonomy', {}).get('topic_path', [])
                print(f"Topic paths: {topic_paths}")

                valid_paths = [path for path in topic_paths if path in validator.valid_topic_paths]
                invalid_paths = [path for path in topic_paths if path not in validator.valid_topic_paths]

                if valid_paths:
                    print(f"✅ Valid paths: {valid_paths}")
                if invalid_paths:
                    print(f"❌ Invalid paths: {invalid_paths}")

        await client.close()
    else:
        print("⚠️ No database connection - skipping database tests")

    # Show some examples of valid enums
    print(f"\n📚 Sample Valid Skill Tags:")
    sample_skills = list(validator.valid_skill_tags)[:20]
    for skill in sample_skills:
        print(f"  - {skill}")

    print(f"\n📚 Sample Valid Subject References:")
    sample_refs = list(validator.valid_subject_refs)[:10]
    for ref in sample_refs:
        print(f"  - {ref}")

    print(f"\n📚 Sample Valid Topic Paths:")
    sample_paths = list(validator.valid_topic_paths)[:15]
    for path in sample_paths:
        print(f"  - {path}")

if __name__ == "__main__":
    asyncio.run(test_validation())
