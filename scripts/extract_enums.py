#!/usr/bin/env python3
"""Extract skill tags and subject content references from official syllabus and past papers"""

import json
import re
import os
import sys

def safe_json_load(file_path: str):
    """Safely load JSON with LaTeX escape handling"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Try normal JSON loading first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print("JSON decode error, attempting to fix LaTeX escapes...")

            # Fix common LaTeX commands that need double backslashes
            latex_commands = [
                'frac', 'text', 'le', 'times', 'cap', 'xi', 'quad', 'circ',
                'ge', 'ne', 'pm', 'sqrt', 'cdot', 'div', 'sum', 'int', 'lim',
                'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'min', 'max'
            ]

            for cmd in latex_commands:
                pattern = rf'(?<!\\)\\{cmd}(?![a-zA-Z])'
                content = re.sub(pattern, rf'\\\\{cmd}', content)

            # Fix other common LaTeX patterns
            content = re.sub(r'(?<!\\)\\(?=[{}\[\]()^_])', r'\\\\', content)
            content = re.sub(r'(?<!\\)\\degree', r'\\\\degree', content)

            return json.loads(content)

    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return {}

def extract_subject_refs_from_syllabus():
    """Extract subject content references from official syllabus (source of truth)"""
    syllabus_path = "data/syllabus_command.json"

    if not os.path.exists(syllabus_path):
        print(f"Syllabus file not found: {syllabus_path}")
        return {}

    print(f"Loading syllabus from {syllabus_path}...")
    syllabus_data = safe_json_load(syllabus_path)

    if not syllabus_data:
        print("Failed to load syllabus data")
        return {}

    subject_refs = {}

    # Extract from core content
    for topic in syllabus_data.get("core_subject_content", []):
        for sub_topic in topic.get("sub_topics", []):
            ref = sub_topic.get("subject_content_ref")
            title = sub_topic.get("title", "")
            if ref:
                subject_refs[ref] = {
                    "title": title,
                    "type": "core",
                    "topic": topic.get("topic_name", "")
                }

    # Extract from extended content
    for topic in syllabus_data.get("extended_subject_content", []):
        for sub_topic in topic.get("sub_topics", []):
            ref = sub_topic.get("subject_content_ref")
            title = sub_topic.get("title", "")
            if ref:
                subject_refs[ref] = {
                    "title": title,
                    "type": "extended",
                    "topic": topic.get("topic_name", "")
                }

    return subject_refs

def extract_skill_tags_from_past_papers():
    """Extract skill tags from past papers (for real-world usage patterns)"""
    file_path = "data/processed/2025p1.json"

    if not os.path.exists(file_path):
        print(f"Past papers file not found: {file_path}")
        return set()

    print(f"Loading past papers from {file_path}...")
    data = safe_json_load(file_path)

    if not data:
        print("Failed to load past papers data")
        return set()

    skill_tags = set()
    topic_paths = set()

    questions = data.get('questions', [])
    print(f"Processing {len(questions)} questions for skill tags...")

    for question in questions:
        taxonomy = question.get('taxonomy', {})

        # Extract skill tags
        tags = taxonomy.get('skill_tags', [])
        if isinstance(tags, list):
            skill_tags.update(tags)
        elif isinstance(tags, str):
            skill_tags.add(tags)

        # Extract topic paths
        paths = taxonomy.get('topic_path', [])
        if isinstance(paths, list):
            for path in paths:
                topic_paths.add(path)

    return skill_tags, topic_paths

def generate_enum_code():
    """Generate the complete enums.py file with proper descriptions"""

    # Get data from source of truth
    subject_refs = extract_subject_refs_from_syllabus()
    skill_tags, topic_paths = extract_skill_tags_from_past_papers()

    print(f"\n📚 Found {len(subject_refs)} subject content references from official syllabus")
    print(f"🏷️ Found {len(skill_tags)} skill tags from past papers")
    print(f"📂 Found {len(topic_paths)} topic path components")

    # Generate the enum file content
    enum_content = '''"""
Enums for Cambridge IGCSE Mathematics question generation.
Based on official Cambridge syllabus (data/syllabus_command.json) and actual past papers.
"""

from enum import Enum


class SkillTag(Enum):
    """Valid skill tags extracted from Cambridge IGCSE Mathematics past papers"""
'''

    # Add skill tags
    for tag in sorted(skill_tags):
        enum_name = tag
        enum_content += f'    {enum_name} = "{tag}"\n'

    enum_content += '''

class SubjectContentReference(Enum):
    """Valid Cambridge IGCSE Mathematics subject content references from official syllabus"""
'''

    # Add subject content references with descriptions
    for ref in sorted(subject_refs.keys()):
        details = subject_refs[ref]
        enum_name = ref.replace(".", "_")
        title = details["title"]
        topic = details["topic"]
        ref_type = details["type"]

        enum_content += f'    {enum_name} = "{ref}"  # {title} ({topic} - {ref_type})\n'

    enum_content += '''

class TopicPathComponent(Enum):
    """Valid topic path components for Cambridge IGCSE Mathematics"""
'''

    # Add topic paths
    for path in sorted(topic_paths):
        enum_name = path.upper().replace(" ", "_").replace(",", "").replace("(", "").replace(")", "").replace("-", "_")
        # Handle special characters
        enum_name = re.sub(r'[^A-Z0-9_]', '_', enum_name)
        enum_name = re.sub(r'_+', '_', enum_name)  # Replace multiple underscores with single
        enum_name = enum_name.strip('_')  # Remove leading/trailing underscores

        enum_content += f'    {enum_name} = "{path}"\n'

    # Add helper functions
    enum_content += '''

# Helper functions to convert strings to enums
def skill_tag_from_string(tag_str: str) -> SkillTag:
    """Convert string to SkillTag enum, with fallback"""
    try:
        return SkillTag(tag_str)
    except ValueError:
        # Fallback for unrecognized skill tags
        print(f"Warning: Unrecognized skill tag '{tag_str}', using WORD_PROBLEM as fallback")
        return SkillTag.WORD_PROBLEM


def subject_ref_from_string(ref_str: str) -> SubjectContentReference:
    """Convert string to SubjectContentReference enum, with fallback"""
    try:
        return SubjectContentReference(ref_str)
    except ValueError:
        # Fallback for unrecognized references
        print(f"Warning: Unrecognized subject reference '{ref_str}', using C1_6 as fallback")
        return SubjectContentReference.C1_6


def topic_path_from_string(path_str: str) -> TopicPathComponent:
    """Convert string to TopicPathComponent enum, with fallback"""
    try:
        return TopicPathComponent(path_str)
    except ValueError:
        # Fallback for unrecognized topic paths
        print(f"Warning: Unrecognized topic path '{path_str}', using NUMBER as fallback")
        return TopicPathComponent.NUMBER


# Validation helpers
def get_valid_skill_tags() -> list[str]:
    """Get list of all valid skill tag strings"""
    return [tag.value for tag in SkillTag]


def get_valid_subject_refs() -> list[str]:
    """Get list of all valid subject content reference strings"""
    return [ref.value for ref in SubjectContentReference]


def get_valid_topic_paths() -> list[str]:
    """Get list of all valid topic path component strings"""
    return [path.value for path in TopicPathComponent]
'''

    return enum_content

def extract_enums():
    """Main function - extract and display enum information"""

    print("🔍 Extracting enums from official sources...")

    # Generate the new enum file content
    enum_content = generate_enum_code()

    # Write to file
    output_path = "src/models/enums.py"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(enum_content)

    print(f"\n✅ Generated complete enums.py file: {output_path}")
    print("📚 Subject content references sourced from official syllabus")
    print("🏷️ Skill tags sourced from actual past papers")
    print("🔗 All references include descriptive comments")

if __name__ == "__main__":
    extract_enums()
