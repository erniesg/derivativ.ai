#!/usr/bin/env python3
"""Test question generation using config file with validation"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.config_manager import ConfigManager
from src.models.question_models import GenerationConfig
from src.models.enums import get_valid_skill_tags, get_valid_subject_refs

def test_config_validation():
    """Test that configs in question_generation_configs.json are validated"""
    print("🧪 Testing Config File Validation...")

    # Load config manager
    config_manager = ConfigManager("config/question_generation_configs.json")

    # Test 1: Check available configs
    available_configs = config_manager.list_available_configs()
    print(f"\n📋 Available configs: {available_configs}")

    # Test 2: Check each config for validation issues
    valid_subject_refs = set(get_valid_subject_refs())

    for config_id in available_configs:
        print(f"\n🔍 Testing config: {config_id}")

        config_template = config_manager.get_config_template(config_id)
        if config_template:
            # Check subject content references
            invalid_refs = [ref for ref in config_template.subject_content_references
                          if ref not in valid_subject_refs]

            if invalid_refs:
                print(f"❌ Invalid subject references: {invalid_refs}")
            else:
                print(f"✅ All subject references valid: {config_template.subject_content_references}")

            # Test creating a GenerationConfig
            try:
                generation_config = config_manager.create_generation_config(config_id, target_grade=5)
                if generation_config:
                    print(f"✅ GenerationConfig created successfully")
                    print(f"   Model: {generation_config.llm_model_generation}")
                    print(f"   Marks: {generation_config.desired_marks}")
                    print(f"   Subject refs: {generation_config.subject_content_references}")
                else:
                    print(f"❌ Failed to create GenerationConfig")
            except Exception as e:
                print(f"❌ Error creating config: {e}")

    # Test 3: Check if we're passing enum data to prompts
    print(f"\n🏷️ Available skill tags: {len(get_valid_skill_tags())} total")
    sample_skills = get_valid_skill_tags()[:10]
    print(f"   Sample: {sample_skills}")

    print(f"\n📚 Available subject refs: {len(get_valid_subject_refs())} total")
    print(f"   All: {get_valid_subject_refs()}")

def test_prompt_skill_tags():
    """Test if our prompts contain current skill tag examples"""
    print("\n🔍 Testing Prompt Skill Tag Coverage...")

    # Load prompt template
    try:
        with open("prompts/question_generation_v1.1.txt", "r") as f:
            prompt_content = f.read()

        valid_skills = get_valid_skill_tags()

        # Check how many of our valid skill tags are mentioned in the prompt
        mentioned_skills = []
        for skill in valid_skills:
            if skill in prompt_content:
                mentioned_skills.append(skill)

        print(f"📊 Skill tags mentioned in prompt: {len(mentioned_skills)}/{len(valid_skills)}")
        print(f"   Mentioned: {mentioned_skills[:10]}...")

        not_mentioned = [skill for skill in valid_skills if skill not in prompt_content]
        print(f"❌ Not mentioned: {len(not_mentioned)} skills")
        if not_mentioned:
            print(f"   Examples: {not_mentioned[:10]}...")

        # Check if prompt has dynamic placeholders
        if "{skill_tags}" in prompt_content:
            print("✅ Prompt has dynamic skill tag placeholder")
        else:
            print("❌ Prompt uses hardcoded skill tag examples")

    except FileNotFoundError:
        print("❌ Prompt template not found")

async def test_end_to_end_config():
    """Test end-to-end generation with config (without actual LLM call)"""
    print("\n🔗 Testing End-to-End Config Flow...")

    config_manager = ConfigManager("config/question_generation_configs.json")

    # Test a specific config
    config_id = "mixed_review_gpt4o_mini"

    print(f"📋 Testing config: {config_id}")
    config_info = config_manager.get_config_info(config_id)

    if config_info:
        print(f"✅ Config loaded successfully:")
        print(f"   Description: {config_info['description']}")
        print(f"   Subject areas: {config_info['subject_areas']}")
        print(f"   Target grades: {config_info['target_grades']}")
        print(f"   Models: {config_info['models']}")

        # Check if subject areas are valid
        valid_refs = set(get_valid_subject_refs())
        invalid_refs = [ref for ref in config_info['subject_areas'] if ref not in valid_refs]

        if invalid_refs:
            print(f"❌ Config has invalid subject references: {invalid_refs}")
        else:
            print(f"✅ All subject references in config are valid")
    else:
        print(f"❌ Failed to load config: {config_id}")

if __name__ == "__main__":
    test_config_validation()
    test_prompt_skill_tags()
    asyncio.run(test_end_to_end_config())
