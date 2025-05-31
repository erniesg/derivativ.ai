#!/usr/bin/env python3
"""Extract skill tags and subject content references from 2025p1.json"""

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

def extract_enums():
    """Extract skill tags and subject content references"""
    file_path = "data/processed/2025p1.json"

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading data from {file_path}...")
    data = safe_json_load(file_path)

    if not data:
        print("Failed to load data")
        return

    skill_tags = set()
    subject_refs = set()
    topic_paths = set()

    questions = data.get('questions', [])
    print(f"Processing {len(questions)} questions...")

    for question in questions:
        taxonomy = question.get('taxonomy', {})

        # Extract skill tags
        tags = taxonomy.get('skill_tags', [])
        if isinstance(tags, list):
            skill_tags.update(tags)
        elif isinstance(tags, str):
            skill_tags.add(tags)

        # Extract subject content refs
        refs = taxonomy.get('subject_content_references', [])
        if isinstance(refs, list):
            subject_refs.update(refs)
        elif isinstance(refs, str):
            subject_refs.add(refs)

        # Extract topic paths
        paths = taxonomy.get('topic_path', [])
        if isinstance(paths, list):
            for path in paths:
                topic_paths.add(path)

    print(f"\n=== SKILL TAGS ({len(skill_tags)} total) ===")
    for tag in sorted(skill_tags):
        print(f'    "{tag}",')

    print(f"\n=== SUBJECT CONTENT REFERENCES ({len(subject_refs)} total) ===")
    for ref in sorted(subject_refs):
        print(f'    "{ref}",')

    print(f"\n=== TOPIC PATH COMPONENTS ({len(topic_paths)} total) ===")
    for path in sorted(topic_paths):
        print(f'    "{path}",')

if __name__ == "__main__":
    extract_enums()
