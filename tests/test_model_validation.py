#!/usr/bin/env python3
"""Test the updated QuestionTaxonomy model validation"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.question_models import QuestionTaxonomy

def test_model_validation():
    """Test model-level validation in QuestionTaxonomy"""
    print("🧪 Testing Updated QuestionTaxonomy Model Validation...")

    # Test 1: Valid data should work
    print("\n1️⃣ Testing valid taxonomy...")
    try:
        valid_taxonomy = QuestionTaxonomy(
            topic_path=['Number', 'Fractions'],
            subject_content_references=['C1.4', 'C1.6'],
            skill_tags=['ADDITION', 'FRACTION_OF_QUANTITY']
        )
        print('✅ Valid taxonomy created successfully')
    except Exception as e:
        print(f'❌ Valid test failed: {e}')

    # Test 2: Invalid subject reference should fail
    print("\n2️⃣ Testing invalid subject reference...")
    try:
        invalid_taxonomy = QuestionTaxonomy(
            topic_path=['Number'],
            subject_content_references=['INVALID_REF', 'C1.4'],
            skill_tags=['ADDITION']
        )
        print('❌ Should have failed for invalid subject ref')
    except ValueError as e:
        print(f'✅ Correctly caught invalid subject ref: {str(e)[:100]}...')
    except Exception as e:
        print(f'⚠️ Unexpected error: {e}')

    # Test 3: Invalid skill tags should warn but not fail
    print("\n3️⃣ Testing invalid skill tags (should warn but allow)...")
    try:
        warning_taxonomy = QuestionTaxonomy(
            topic_path=['Number'],
            subject_content_references=['C1.4'],
            skill_tags=['INVALID_SKILL', 'ADDITION']
        )
        print('✅ Warning case handled (non-standard skill tags allowed with warning)')
    except Exception as e:
        print(f'❌ Unexpected failure: {e}')

    # Test 4: Show valid options
    print("\n4️⃣ Valid options summary:")
    from src.models.enums import get_valid_skill_tags, get_valid_subject_refs
    valid_skills = get_valid_skill_tags()
    valid_refs = get_valid_subject_refs()

    print(f"📊 Available: {len(valid_skills)} skill tags, {len(valid_refs)} subject refs")
    print(f"🏷️ Example skill tags: {valid_skills[:5]}")
    print(f"📚 Example subject refs: {valid_refs[:5]}")

if __name__ == "__main__":
    test_model_validation()
