#!/usr/bin/env python3
"""Test contextual skill tag injection vs global injection"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.prompt_loader import PromptLoader
from src.services.skill_tag_mapper import SkillTagMapper
from src.models.enums import get_valid_skill_tags

def test_contextual_vs_global_skill_tags():
    """Test the difference between contextual and global skill tag injection"""
    print("🧪 Testing Contextual vs Global Skill Tag Injection...")

    prompt_loader = PromptLoader()
    skill_mapper = SkillTagMapper()

    # Test cases with different subject content references
    test_cases = [
        {
            "name": "Basic Arithmetic",
            "subject_refs": ["C1.1", "C1.6"],  # Number topics
            "description": "Types of number, The four operations"
        },
        {
            "name": "Algebra",
            "subject_refs": ["C2.2", "C2.5"],  # Algebra topics
            "description": "Algebraic manipulation, Equations"
        },
        {
            "name": "Geometry",
            "subject_refs": ["C4.1", "C4.6"],  # Geometry topics
            "description": "Geometrical terms, Angles"
        },
        {
            "name": "Probability",
            "subject_refs": ["C8.3"],  # Probability topics
            "description": "Probability of combined events"
        },
        {
            "name": "Mixed Topics",
            "subject_refs": ["C1.6", "C2.2", "C4.1"],  # Mixed
            "description": "Four operations, Algebraic manipulation, Geometrical terms"
        }
    ]

    all_skill_tags = get_valid_skill_tags()
    print(f"\n📊 Total skill tags available: {len(all_skill_tags)}")

    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"🔍 Test Case: {test_case['name']}")
        print(f"📚 Subject Refs: {test_case['subject_refs']}")
        print(f"📝 Description: {test_case['description']}")

        # Get contextual skill tags
        contextual_tags = skill_mapper.get_relevant_skill_tags(test_case['subject_refs'])

        print(f"\n📋 Results:")
        print(f"   🎯 Contextual tags: {len(contextual_tags)}/{len(all_skill_tags)} ({len(contextual_tags)/len(all_skill_tags)*100:.1f}%)")
        print(f"   🌍 Global tags: {len(all_skill_tags)}/97 (100%)")
        print(f"   📉 Reduction: {len(all_skill_tags) - len(contextual_tags)} tags removed")

        # Show specific relevant tags
        print(f"\n🏷️ Contextual skill tags for {test_case['name']}:")
        for tag in contextual_tags[:15]:  # Show first 15
            print(f"   • {tag}")
        if len(contextual_tags) > 15:
            print(f"   ... and {len(contextual_tags) - 15} more")

        # Show some irrelevant tags that would be excluded
        irrelevant_tags = [tag for tag in all_skill_tags if tag not in contextual_tags]
        if irrelevant_tags:
            print(f"\n❌ Example irrelevant tags excluded:")
            for tag in irrelevant_tags[:8]:  # Show first 8
                print(f"   • {tag}")
            if len(irrelevant_tags) > 8:
                print(f"   ... and {len(irrelevant_tags) - 8} more excluded")

def test_prompt_generation_with_contextual_tags():
    """Test actual prompt generation with contextual skill tags"""
    print(f"\n{'='*60}")
    print("🚀 Testing Prompt Generation with Contextual Skill Tags...")

    prompt_loader = PromptLoader()

    # Test context for algebra questions
    algebra_context = {
        'target_grade': 5,
        'calculator_policy': 'allowed',
        'desired_marks': 2,
        'subject_content_references': '["C2.2", "C2.5"]',  # Algebra topics
        'command_word': 'Solve',
        'syllabus_content': 'Algebraic manipulation and equations',
        'command_word_definition': 'work out from given facts',
        'seed_question_context': 'No seed question',
        'marking_principles': 'Test principles',
        'generation_id': 'test-algebra-123'
    }

    try:
        algebra_prompt = prompt_loader.format_question_generation_prompt("v1.2", **algebra_context)

        # Count skill tags in the prompt
        from src.models.enums import get_valid_skill_tags
        all_tags = get_valid_skill_tags()

        tags_in_prompt = []
        for tag in all_tags:
            if tag in algebra_prompt:
                tags_in_prompt.append(tag)

        print(f"✅ Algebra prompt generated successfully")
        print(f"🏷️ Skill tags in prompt: {len(tags_in_prompt)}/{len(all_tags)}")

        # Show which algebra-relevant tags are included
        algebra_relevant = ["SUBSTITUTION", "SOLVE_LINEAR_EQUATION", "FORM_EQUATION",
                          "FACTORISATION_COMMON_FACTOR", "SIMULTANEOUS_LINEAR_EQUATIONS"]
        included_relevant = [tag for tag in algebra_relevant if tag in tags_in_prompt]
        print(f"✅ Relevant algebra tags included: {included_relevant}")

        # Show which irrelevant tags are excluded
        irrelevant_examples = ["TREE_DIAGRAM_USE", "ALTERNATE_ANGLES", "BEARINGS", "TRANSLATION"]
        excluded_irrelevant = [tag for tag in irrelevant_examples if tag not in tags_in_prompt]
        print(f"❌ Irrelevant tags excluded: {excluded_irrelevant}")

        # Show a sample of the skill tags section
        skill_section_start = algebra_prompt.find("**Algebra:**")
        if skill_section_start != -1:
            skill_section_end = algebra_prompt.find("\n\n**", skill_section_start + 1)
            if skill_section_end == -1:
                skill_section_end = skill_section_start + 200
            sample = algebra_prompt[skill_section_start:skill_section_end]
            print(f"\n📝 Sample of algebra skill tags section:")
            print(f"   {sample}")

    except Exception as e:
        print(f"❌ Error generating prompt: {e}")

def test_mapping_coverage():
    """Test coverage of skill tag mappings"""
    print(f"\n{'='*60}")
    print("📊 Testing Skill Tag Mapping Coverage...")

    skill_mapper = SkillTagMapper()
    all_skill_tags = set(get_valid_skill_tags())

    # Get all mapped skill tags
    mappings = skill_mapper.get_topic_specific_mappings()
    mapped_tags = set()

    for topic, tags in mappings.items():
        mapped_tags.update(tags)
        print(f"🗂️ {topic}: {len(tags)} skill tags")

    # Check coverage
    unmapped_tags = all_skill_tags - mapped_tags
    mapped_count = len(mapped_tags & all_skill_tags)

    print(f"\n📈 Coverage Analysis:")
    print(f"   ✅ Mapped skill tags: {mapped_count}/{len(all_skill_tags)} ({mapped_count/len(all_skill_tags)*100:.1f}%)")

    if unmapped_tags:
        print(f"   ⚠️ Unmapped skill tags: {len(unmapped_tags)}")
        for tag in sorted(unmapped_tags)[:10]:
            print(f"      • {tag}")
        if len(unmapped_tags) > 10:
            print(f"      ... and {len(unmapped_tags) - 10} more")
    else:
        print(f"   🎉 All skill tags are mapped!")

if __name__ == "__main__":
    test_contextual_vs_global_skill_tags()
    test_prompt_generation_with_contextual_tags()
    test_mapping_coverage()
