#!/usr/bin/env python3
"""Test the new v1.2 prompt template with dynamic skill tag injection"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.prompt_loader import PromptLoader
from src.models.enums import get_valid_skill_tags

def test_v1_2_dynamic_skill_tags():
    """Test that v1.2 template uses dynamic skill tag injection"""
    print("🧪 Testing v1.2 Prompt Template with Dynamic Skill Tags...")

    prompt_loader = PromptLoader()

    # Test context variables
    context = {
        'target_grade': 5,
        'calculator_policy': 'allowed',
        'desired_marks': 2,
        'subject_content_references': '["C2.2", "C2.5"]',
        'command_word': 'Calculate',
        'syllabus_content': 'Test syllabus content',
        'command_word_definition': 'work out from given facts',
        'seed_question_context': 'No seed question',
        'marking_principles': 'Test principles',
        'generation_id': 'test-123'
    }

    # Test v1.1 (old template with hardcoded skill tags)
    print("\n1️⃣ Testing v1.1 template (hardcoded skill tags)...")
    try:
        v1_1_prompt = prompt_loader.format_question_generation_prompt("v1.1", **context)

        # Check if it has hardcoded skill tags
        hardcoded_count = 0
        for skill in ["ADDITION", "SUBTRACTION", "ALTERNATE_ANGLES", "CONSTRUCTION"]:
            if skill in v1_1_prompt:
                hardcoded_count += 1

        print(f"   ✅ v1.1 loaded successfully")
        print(f"   📊 Found {hardcoded_count} hardcoded skill tag examples")
        print(f"   🔍 Contains placeholder: {'{{skill_tags}}' in v1_1_prompt}")

    except Exception as e:
        print(f"   ❌ Error with v1.1: {e}")

    # Test v1.2 (new template with dynamic injection)
    print("\n2️⃣ Testing v1.2 template (dynamic skill tags)...")
    try:
        v1_2_prompt = prompt_loader.format_question_generation_prompt("v1.2", **context)

        # Check if all skill tags are present
        all_skill_tags = get_valid_skill_tags()
        missing_skills = []
        present_skills = []

        for skill in all_skill_tags:
            if skill in v1_2_prompt:
                present_skills.append(skill)
            else:
                missing_skills.append(skill)

        print(f"   ✅ v1.2 loaded successfully")
        print(f"   📊 Skill tags present: {len(present_skills)}/{len(all_skill_tags)}")
        print(f"   🔍 Contains placeholder: {'{{skill_tags}}' in v1_2_prompt}")

        if len(present_skills) == len(all_skill_tags):
            print(f"   🎉 SUCCESS: All {len(all_skill_tags)} skill tags dynamically injected!")
        else:
            print(f"   ⚠️ Missing {len(missing_skills)} skill tags: {missing_skills[:5]}...")

        # Check formatting
        if "**Number & Arithmetic:**" in v1_2_prompt:
            print(f"   ✅ Skill tags properly categorized")
        else:
            print(f"   ❌ Skill tags not properly categorized")

        # Show sample of formatted content
        skill_section_start = v1_2_prompt.find("**Number & Arithmetic:**")
        if skill_section_start != -1:
            sample = v1_2_prompt[skill_section_start:skill_section_start + 200]
            print(f"   📝 Sample formatting: {sample}...")

    except Exception as e:
        print(f"   ❌ Error with v1.2: {e}")

def test_skill_tag_categorization():
    """Test the skill tag categorization logic"""
    print("\n🗂️ Testing Skill Tag Categorization...")

    prompt_loader = PromptLoader()
    all_skills = get_valid_skill_tags()

    formatted = prompt_loader._format_skill_tags_for_prompt(all_skills)

    categories = [
        "**Number & Arithmetic:**",
        "**Algebra:**",
        "**Geometry:**",
        "**Statistics & Probability:**",
        "**Transformations:**",
        "**General:**"
    ]

    found_categories = []
    for category in categories:
        if category in formatted:
            found_categories.append(category.replace("**", "").replace(":", ""))

    print(f"   📂 Categories found: {len(found_categories)}/{len(categories)}")
    print(f"   📋 Categories: {', '.join(found_categories)}")

    # Count skills in each category
    for category in categories:
        if category in formatted:
            # Find the section for this category
            start = formatted.find(category)
            next_category_start = len(formatted)

            for other_cat in categories:
                other_start = formatted.find(other_cat, start + 1)
                if other_start != -1 and other_start < next_category_start:
                    next_category_start = other_start

            section = formatted[start:next_category_start]
            skill_count = section.count(',') + 1 if ',' in section else (1 if any(skill in section for skill in all_skills) else 0)
            print(f"   📊 {category}: ~{skill_count} skills")

if __name__ == "__main__":
    test_v1_2_dynamic_skill_tags()
    test_skill_tag_categorization()
