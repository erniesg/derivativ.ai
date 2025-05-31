#!/usr/bin/env python3
"""Test question generation with contextual skill tags using actual configs"""

import sys
import os
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.config_manager import ConfigManager
from src.services.prompt_loader import PromptLoader
from src.services.skill_tag_mapper import SkillTagMapper
from src.models.enums import get_valid_skill_tags, get_valid_subject_refs

def test_configs_with_contextual_skill_tags():
    """Test all configs with new contextual skill tag injection"""
    print("🧪 Testing Question Generation Configs with Contextual Skill Tags...")

    config_manager = ConfigManager()
    prompt_loader = PromptLoader()
    skill_mapper = SkillTagMapper()

    # Load all configs
    all_configs = config_manager.list_available_configs()
    print(f"\n📋 Testing {len(all_configs)} configurations...")

    all_skill_tags = get_valid_skill_tags()

    for config_id in all_configs:
        print(f"\n{'='*70}")
        print(f"🔍 Testing Config: {config_id}")

        # Get config template
        config_template = config_manager.get_config_template(config_id)
        if not config_template:
            print(f"❌ Could not load config template for {config_id}")
            continue

        print(f"📝 Description: {config_template.description}")
        print(f"📚 Subject refs: {config_template.subject_content_references}")
        print(f"🎯 Target grades: {config_template.target_grades}")
        print(f"🤖 Generation model: {config_template.llm_model_generation}")
        print(f"📄 Prompt version: {config_template.prompt_template_version_generation}")

        # Test contextual skill tag mapping
        relevant_skills = skill_mapper.get_relevant_skill_tags(config_template.subject_content_references)
        reduction_percentage = (1 - len(relevant_skills) / len(all_skill_tags)) * 100

        print(f"\n🏷️ Skill Tag Analysis:")
        print(f"   🎯 Contextual tags: {len(relevant_skills)}/{len(all_skill_tags)} ({len(relevant_skills)/len(all_skill_tags)*100:.1f}%)")
        print(f"   📉 Reduction: {reduction_percentage:.1f}% of irrelevant tags excluded")

        # Show sample of relevant tags
        print(f"   🔖 Sample relevant tags: {relevant_skills[:8]}...")

        # Show some excluded tags
        excluded_tags = [tag for tag in all_skill_tags if tag not in relevant_skills]
        if excluded_tags:
            print(f"   ❌ Sample excluded tags: {excluded_tags[:6]}...")

        # Test prompt generation with contextual tags
        if config_template.prompt_template_version_generation >= "v1.2":
            print(f"\n🚀 Testing contextual prompt generation (v{config_template.prompt_template_version_generation})...")
            test_prompt_generation(config_id, config_manager, prompt_loader)
        else:
            print(f"\n⚠️ Prompt version {config_template.prompt_template_version_generation} uses hardcoded skill tags")

def test_prompt_generation(config_id: str, config_manager: ConfigManager, prompt_loader: PromptLoader):
    """Test prompt generation for a specific config"""
    try:
        # Create a generation config
        generation_config = config_manager.create_generation_config(config_id, target_grade=5)
        if not generation_config:
            print(f"   ❌ Could not create generation config")
            return

        # Prepare context for prompt generation
        context = {
            'target_grade': generation_config.target_grade,
            'calculator_policy': generation_config.calculator_policy,
            'desired_marks': generation_config.desired_marks,
            'subject_content_references': json.dumps(generation_config.subject_content_references),
            'command_word': 'Calculate',
            'syllabus_content': 'Test syllabus content',
            'command_word_definition': 'work out from given facts',
            'seed_question_context': 'No seed question',
            'marking_principles': 'Test principles',
            'generation_id': f'test-{config_id}-123'
        }

        # Generate prompt
        prompt = prompt_loader.format_question_generation_prompt(
            generation_config.prompt_template_version_generation,
            **context
        )

        # Analyze the generated prompt
        skill_tags_in_prompt = []
        all_tags = get_valid_skill_tags()

        for tag in all_tags:
            if tag in prompt:
                skill_tags_in_prompt.append(tag)

        print(f"   ✅ Prompt generated successfully")
        print(f"   🏷️ Skill tags in prompt: {len(skill_tags_in_prompt)}/{len(all_tags)}")

        # Check if contextual tags are properly injected
        skill_mapper = SkillTagMapper()
        relevant_tags = skill_mapper.get_relevant_skill_tags(generation_config.subject_content_references)

        relevant_in_prompt = [tag for tag in relevant_tags if tag in skill_tags_in_prompt]
        print(f"   ✅ Relevant tags included: {len(relevant_in_prompt)}/{len(relevant_tags)} ({len(relevant_in_prompt)/len(relevant_tags)*100:.1f}%)")

        # Check for properly categorized sections
        categories_found = []
        categories = ["**Number & Arithmetic:**", "**Algebra:**", "**Geometry:**",
                     "**Statistics & Probability:**", "**Transformations:**", "**General:**"]

        for category in categories:
            if category in prompt:
                categories_found.append(category.replace("**", "").replace(":", ""))

        if categories_found:
            print(f"   📂 Categories found: {', '.join(categories_found)}")

    except Exception as e:
        print(f"   ❌ Error generating prompt: {e}")

def test_specific_config_scenarios():
    """Test specific scenarios with different config types"""
    print(f"\n{'='*70}")
    print("🎯 Testing Specific Config Scenarios...")

    scenarios = [
        {
            "config_id": "basic_arithmetic_gpt4o",
            "expected_categories": ["Number & Arithmetic"],
            "expected_skills": ["ADDITION", "SUBTRACTION", "MULTIPLICATION"],
            "should_exclude": ["ALTERNATE_ANGLES", "TREE_DIAGRAM_USE", "ROTATION"]
        },
        {
            "config_id": "algebra_claude4",
            "expected_categories": ["Algebra"],
            "expected_skills": ["SUBSTITUTION", "SOLVE_LINEAR_EQUATION", "FORM_EQUATION"],
            "should_exclude": ["BEARINGS", "TREE_DIAGRAM_USE", "AREA_COMPOSITE_SHAPES"]
        },
        {
            "config_id": "geometry_gemini",
            "expected_categories": ["Geometry"],
            "expected_skills": ["ANGLE_PROPERTIES", "CONSTRUCTION", "SCALE_DRAWING"],
            "should_exclude": ["PROBABILITY_COMPLEMENT", "SUBSTITUTION", "TIME_CALCULATION"]
        },
        {
            "config_id": "probability_qwen",
            "expected_categories": ["Statistics & Probability"],
            "expected_skills": ["PROBABILITY_COMPLEMENT", "TREE_DIAGRAM_USE"],
            "should_exclude": ["ALTERNATE_ANGLES", "SUBSTITUTION", "CONSTRUCTION"]
        }
    ]

    config_manager = ConfigManager()
    skill_mapper = SkillTagMapper()

    for scenario in scenarios:
        print(f"\n🔍 Scenario: {scenario['config_id']}")

        config_template = config_manager.get_config_template(scenario['config_id'])
        if not config_template:
            print(f"   ❌ Config not found")
            continue

        # Get contextual tags for this config
        relevant_tags = skill_mapper.get_relevant_skill_tags(config_template.subject_content_references)

        # Check expected skills are included
        included_expected = [skill for skill in scenario['expected_skills'] if skill in relevant_tags]
        print(f"   ✅ Expected skills included: {included_expected}")

        # Check excluded skills are properly excluded
        excluded_correctly = [skill for skill in scenario['should_exclude'] if skill not in relevant_tags]
        print(f"   ❌ Correctly excluded: {excluded_correctly}")

        # Summary
        if len(included_expected) == len(scenario['expected_skills']) and len(excluded_correctly) == len(scenario['should_exclude']):
            print(f"   🎉 Perfect contextual filtering!")
        else:
            print(f"   ⚠️ Some issues with filtering detected")

def test_mixed_topic_configs():
    """Test configs with mixed topics to see how skill tag union works"""
    print(f"\n{'='*70}")
    print("🔀 Testing Mixed Topic Configurations...")

    config_manager = ConfigManager()
    skill_mapper = SkillTagMapper()

    # Test mixed_review_gpt4o_mini config
    config_template = config_manager.get_config_template("mixed_review_gpt4o_mini")
    if config_template:
        print(f"\n🔍 Mixed Config: mixed_review_gpt4o_mini")
        print(f"📚 Subject refs: {config_template.subject_content_references}")

        # Get contextual tags
        relevant_tags = skill_mapper.get_relevant_skill_tags(config_template.subject_content_references)

        # Analyze which topic areas contribute
        topic_contributions = {}

        for ref in config_template.subject_content_references:
            single_ref_tags = skill_mapper.get_relevant_skill_tags([ref])
            topic_area = ref.split('.')[0]  # C1, C2, etc.

            topic_map = {
                'C1': 'Number', 'C2': 'Algebra', 'C3': 'Coordinate Geometry',
                'C4': 'Geometry', 'C5': 'Mensuration', 'C6': 'Trigonometry',
                'C7': 'Transformations', 'C8': 'Probability', 'C9': 'Statistics'
            }

            topic_name = topic_map.get(topic_area, topic_area)
            topic_contributions[topic_name] = len(single_ref_tags)

        print(f"🗂️ Topic contributions:")
        for topic, count in topic_contributions.items():
            print(f"   • {topic}: {count} skill tags")

        print(f"🏷️ Total contextual tags: {len(relevant_tags)} (union of all topic areas)")

        # Show sample from each category
        categories = ["Number & Arithmetic", "Algebra", "Geometry", "Statistics & Probability"]
        for category in categories:
            category_tags = [tag for tag in relevant_tags if any(x in tag.lower() for x in category.lower().split(' & '))]
            if category_tags:
                print(f"   📂 {category}: {len(category_tags)} tags, sample: {category_tags[:3]}")

if __name__ == "__main__":
    test_configs_with_contextual_skill_tags()
    test_specific_config_scenarios()
    test_mixed_topic_configs()
