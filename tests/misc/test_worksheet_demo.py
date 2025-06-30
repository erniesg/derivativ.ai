#!/usr/bin/env python3
"""Test worksheet generation with student/teacher versions."""

import requests
from dotenv import load_dotenv

load_dotenv()


def generate_worksheet():  # noqa: PLR0915
    """Generate a complete worksheet with worked solutions."""
    print("🎯 Generating Complete Algebra Worksheet Demo\n")

    api_url = "http://localhost:8000/api/generation/documents/generate-v2"

    # Create a comprehensive worksheet request
    request_data = {
        "title": "Linear Equations and Graphs Practice",
        "document_type": "worksheet",
        "topic": "Algebra and graphs",
        "tier": "Core",
        "grade_level": 8,
        "target_time_minutes": 45,
        "detail_level": 7,  # Higher detail for more content
        "custom_instructions": (
            "Create a comprehensive worksheet with:\n"
            "1. Clear learning objectives\n"
            "2. Worked examples showing step-by-step solutions\n"
            "3. Practice questions with varying difficulty\n"
            "4. Include practical real-world applications\n"
            "5. Provide full worked solutions for all questions"
        ),
        "force_include_blocks": [
            "learning_objectives",
            "worked_example",
            "practice_questions",
            "summary",
        ],
    }

    print(f"📤 Sending request for: {request_data['title']}")
    print(f"   Detail Level: {request_data['detail_level']}/10")
    print(f"   Target Time: {request_data['target_time_minutes']} minutes")
    print(f"   Blocks: {', '.join(request_data['force_include_blocks'])}\n")

    try:
        # Make API request
        response = requests.post(
            api_url, json=request_data, headers={"Content-Type": "application/json"}, timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            document = result["document"]

            print("✅ Generation Successful!")
            print(f"   Document ID: {document['document_id']}")
            print(f"   Processing Time: {result['processing_time']:.2f}s")
            print(f"   Blocks Generated: {len(document['content_structure']['blocks'])}")

            # Display generated content
            print("\n📄 GENERATED WORKSHEET CONTENT:")
            print("=" * 60)

            # Title
            title = document.get("enhanced_title") or document["title"]
            print(f"\n{title}")
            print("-" * len(title))

            # Introduction
            if document["content_structure"].get("introduction"):
                print(f"\n{document['content_structure']['introduction']}")

            # Process each block
            for i, block in enumerate(document["content_structure"]["blocks"]):
                print(f"\n\n{'='*60}")
                print(f"SECTION {i+1}: {block['block_type'].upper().replace('_', ' ')}")
                print(f"Time: {block['estimated_minutes']} minutes")
                print("=" * 60)

                content = block["content"]

                if block["block_type"] == "learning_objectives":
                    print("\nBy the end of this worksheet, you will be able to:")
                    for obj in content.get("objectives", []):
                        print(f"  • {obj}")

                elif block["block_type"] == "worked_example":
                    print("\nWORKED EXAMPLES:")
                    for j, example in enumerate(content.get("examples", []), 1):
                        print(f"\nExample {j}: {example.get('problem', '')}")
                        print("Solution:")
                        for step in example.get("solution_steps", []):
                            print(
                                f"  Step {step.get('step_number', '')}: {step.get('description', '')}"
                            )
                            if step.get("calculation"):
                                print(f"         {step['calculation']}")
                        print(f"  Answer: {example.get('answer', '')}")

                elif block["block_type"] == "practice_questions":
                    print("\nPRACTICE QUESTIONS:")
                    for j, q in enumerate(content.get("questions", []), 1):
                        print(f"\n{j}. {q.get('text', '')} [{q.get('marks', 0)} marks]")
                        if q.get("hint"):
                            print(f"   Hint: {q['hint']}")
                        if q.get("answer"):
                            print(f"   Answer: {q['answer']}")
                        if q.get("worked_solution"):
                            print(f"   Solution: {q['worked_solution']}")

                elif block["block_type"] == "summary":
                    print("\nKEY TAKEAWAYS:")
                    for point in content.get("key_points", []):
                        print(f"  • {point}")

                    if content.get("tips"):
                        print("\nStudy Tips:")
                        for tip in content["tips"]:
                            print(f"  • {tip}")

            # Coverage notes
            if document["content_structure"].get("coverage_notes"):
                print(f"\n\nCoverage Notes: {document['content_structure']['coverage_notes']}")

            # Storage info
            storage_id = result.get("generation_insights", {}).get("document_id")
            if storage_id:
                print(f"\n\n💾 Document saved to storage with ID: {storage_id}")

            # Show how to create student/teacher versions
            print("\n\n📋 VERSION GENERATION:")
            print("=" * 60)
            print("This document can be rendered in different versions:")
            print("  1. STUDENT VERSION: Questions only, no answers/solutions")
            print("  2. TEACHER VERSION: Complete with all answers and worked solutions")
            print("  3. ANSWER KEY: Just the answers for quick marking")

            return document

        else:
            print(f"❌ Generation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None

    except Exception as e:
        print(f"💥 Error: {e}")
        return None


if __name__ == "__main__":
    generate_worksheet()
