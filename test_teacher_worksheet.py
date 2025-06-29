#!/usr/bin/env python3
"""Generate a teacher version worksheet with full solutions."""

import requests
from dotenv import load_dotenv

load_dotenv()


def generate_teacher_worksheet():  # noqa: PLR0915
    """Generate a teacher version with complete solutions."""
    print("👩‍🏫 Generating Teacher Version Worksheet with Solutions\n")

    api_url = "http://localhost:8000/api/generation/documents/generate-v2"

    request_data = {
        "title": "Quadratic Equations - Teacher Guide",
        "document_type": "worksheet",
        "topic": "Algebra and graphs",
        "tier": "Extended",  # Higher tier for more challenging content
        "grade_level": 9,
        "target_time_minutes": 60,
        "detail_level": 8,  # High detail for comprehensive coverage
        "teacher_version": True,  # Enable teacher version
        "include_answers": True,  # Include all answers
        "custom_instructions": (
            "Create a comprehensive teacher guide with:\n"
            "1. Detailed learning objectives with curriculum links\n"
            "2. Multiple worked examples with step-by-step solutions\n"
            "3. Practice questions with full worked solutions\n"
            "4. Common student misconceptions and how to address them\n"
            "5. Extension activities for advanced students\n"
            "Focus on quadratic equations: factoring, completing the square, and graphing parabolas."
        ),
        "force_include_blocks": [
            "learning_objectives",
            "concept_explanation",
            "worked_example",
            "practice_questions",
            "quick_reference",
            "summary",
        ],
    }

    print(f"📚 Generating: {request_data['title']}")
    print(f"   Grade: {request_data['grade_level']} ({request_data['tier']} tier)")
    print(f"   Duration: {request_data['target_time_minutes']} minutes")
    print(f"   Teacher Version: {request_data['teacher_version']}")
    print(f"   Detail Level: {request_data['detail_level']}/10\n")

    try:
        response = requests.post(
            api_url, json=request_data, headers={"Content-Type": "application/json"}, timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            document = result["document"]

            print("✅ Generation Successful!")
            print(f"   Processing Time: {result['processing_time']:.2f}s")
            print(f"   Blocks Generated: {len(document['content_structure']['blocks'])}\n")

            # Display teacher guide content
            print("=" * 70)
            print("TEACHER GUIDE: " + document["title"])
            print("=" * 70)

            # Show enhanced content
            for block in document["content_structure"]["blocks"]:
                print(f"\n{'='*70}")
                print(f"{block['block_type'].upper().replace('_', ' ')}")
                print(f"Estimated Time: {block['estimated_minutes']} minutes")
                print("=" * 70)

                content = block["content"]

                if block["block_type"] == "concept_explanation":
                    if "title" in content:
                        print(f"\n{content['title']}")
                    if "introduction" in content:
                        print(f"\n{content['introduction']}")

                    for concept in content.get("concepts", []):
                        print(f"\n📌 {concept.get('name', 'Concept')}:")
                        print(f"   {concept.get('explanation', '')}")
                        if "example" in concept:
                            print(f"   Example: {concept['example']}")

                elif block["block_type"] == "practice_questions":
                    print("\nQUESTIONS WITH FULL SOLUTIONS:")
                    for i, q in enumerate(content.get("questions", []), 1):
                        print(f"\nQ{i}. {q.get('text', '')} [{q.get('marks', 0)} marks]")
                        print(f"     Difficulty: {q.get('difficulty', 'medium')}")
                        if q.get("hint"):
                            print(f"     💡 Hint: {q['hint']}")
                        if q.get("answer"):
                            print(f"     ✓ Answer: {q['answer']}")
                        if q.get("worked_solution"):
                            print(f"     📝 Full Solution:\n     {q['worked_solution']}")

                elif block["block_type"] == "quick_reference":
                    if "formulas" in content:
                        print("\nKEY FORMULAS:")
                        for formula in content["formulas"]:
                            print(f"  • {formula}")
                    if "definitions" in content:
                        print("\nIMPORTANT DEFINITIONS:")
                        for term, definition in content.get("definitions", {}).items():
                            print(f"  • {term}: {definition}")

            # Teaching notes
            print("\n" + "=" * 70)
            print("TEACHING NOTES:")
            print("=" * 70)
            print("This worksheet is designed for Extended tier Grade 9 students.")
            print("Adjust pacing based on student understanding.")
            print("\nDifferentiation suggestions:")
            print("  • Core students: Focus on factoring and basic graphing")
            print("  • Extended students: Include completing the square method")
            print("  • Advanced: Add discriminant analysis and transformations")

            return document

        else:
            print(f"❌ Generation failed: {response.status_code}")
            print(f"   Error: {response.text}")

    except Exception as e:
        print(f"💥 Error: {e}")


if __name__ == "__main__":
    generate_teacher_worksheet()
