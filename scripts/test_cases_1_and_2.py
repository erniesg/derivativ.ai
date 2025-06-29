#!/usr/bin/env python3
"""
Quick test of cases 1 and 2 from the E2E test to generate documents in Supabase.
"""

import json
import time
from datetime import datetime

import requests


def test_case_1_algebra_worksheet():
    """Test Case 1: Algebra Worksheet with DB Questions"""
    print("📋 Test Case 1: Algebra Worksheet with DB Questions")

    api_url = "http://localhost:8000/api/generation/documents/generate-v2"

    test_data = {
        "title": "Linear Equations Practice Worksheet",
        "document_type": "worksheet",
        "topic": "Algebra and graphs",  # TopicName.ALGEBRA_AND_GRAPHS.value
        "tier": "Core",
        "grade_level": 8,
        "target_time_minutes": 30,
        "detail_level": 5,
        "custom_instructions": "Focus on solving linear equations with practical applications",
    }

    print(f"Request: {json.dumps(test_data, indent=2)}")
    start_time = time.time()

    try:
        response = requests.post(
            api_url, json=test_data, headers={"Content-Type": "application/json"}, timeout=60
        )

        processing_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS ({processing_time:.2f}s)")

            # Debug: print the full response structure
            print(f"   Raw response keys: {list(result.keys())}")
            print(f"   Success: {result.get('success', False)}")
            print(f"   Error message: {result.get('error_message', 'None')}")

            # Handle different response structures safely
            document = result.get("document")
            if document:
                print(f"   Document ID: {document.get('document_id', 'N/A')}")
                print(f"   Title: {document.get('title', 'N/A')}")

                content_blocks = document.get("content_blocks", [])
                print(f"   Content Blocks: {len(content_blocks)}")

                # Show first few blocks
                for i, block in enumerate(content_blocks[:3]):
                    block_type = block.get("block_type", "Unknown")
                    block_title = block.get("title", "Untitled")
                    print(f"   Block {i+1}: {block_type} - {block_title}")
            else:
                print("   ⚠️ No document in response")
                print(f"   Raw document field: {result.get('document')}")

            processing_time_api = result.get("processing_time", 0)
            if processing_time_api:
                print(f"   Processing Time: {processing_time_api:.2f}s")

            # Check if document was saved to storage
            storage_id = result.get("document_id")
            if storage_id:
                print(f"   💾 SAVED TO SUPABASE: {storage_id}")
                print("   📊 Check stored_documents table for this ID")
            else:
                print("   ⚠️ Not saved to storage")

            return result
        else:
            print(f"❌ FAILED ({response.status_code})")
            print(f"   Error: {response.text}")
            return None

    except Exception as e:
        print(f"💥 ERROR: {e}")
        return None


def test_case_2_geometry_notes():
    """Test Case 2: Geometry Notes with Syllabus References"""
    print("\n📋 Test Case 2: Geometry Notes with Syllabus References")

    api_url = "http://localhost:8000/api/generation/documents/generate-v2"

    test_data = {
        "title": "Coordinate Geometry Study Notes",
        "document_type": "notes",
        "topic": "Coordinate geometry",  # TopicName.COORDINATE_GEOMETRY.value
        "tier": "Extended",
        "grade_level": 9,
        "target_time_minutes": 45,
        "detail_level": 7,
        "subject_content_refs": ["C6.1", "C6.2"],  # Will be converted to enum
        "custom_instructions": "Include step-by-step worked examples and practice problems",
    }

    print(f"Request: {json.dumps(test_data, indent=2)}")
    start_time = time.time()

    try:
        response = requests.post(
            api_url, json=test_data, headers={"Content-Type": "application/json"}, timeout=60
        )

        processing_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS ({processing_time:.2f}s)")

            # Debug: print the full response structure
            print(f"   Raw response keys: {list(result.keys())}")
            print(f"   Success: {result.get('success', False)}")
            print(f"   Error message: {result.get('error_message', 'None')}")

            # Handle different response structures safely
            document = result.get("document")
            if document:
                print(f"   Document ID: {document.get('document_id', 'N/A')}")
                print(f"   Title: {document.get('title', 'N/A')}")

                content_blocks = document.get("content_blocks", [])
                print(f"   Content Blocks: {len(content_blocks)}")

                # Show first few blocks
                for i, block in enumerate(content_blocks[:3]):
                    block_type = block.get("block_type", "Unknown")
                    block_title = block.get("title", "Untitled")
                    print(f"   Block {i+1}: {block_type} - {block_title}")
            else:
                print("   ⚠️ No document in response")
                print(f"   Raw document field: {result.get('document')}")

            processing_time_api = result.get("processing_time", 0)
            if processing_time_api:
                print(f"   Processing Time: {processing_time_api:.2f}s")

            # Check if document was saved to storage
            storage_id = result.get("document_id")
            if storage_id:
                print(f"   💾 SAVED TO SUPABASE: {storage_id}")
                print("   📊 Check stored_documents table for this ID")
            else:
                print("   ⚠️ Not saved to storage")

            return result
        else:
            print(f"❌ FAILED ({response.status_code})")
            print(f"   Error: {response.text}")
            return None

    except Exception as e:
        print(f"💥 ERROR: {e}")
        return None


def check_supabase_tables():
    """Show information about expected Supabase tables"""
    print("\n📊 Supabase Table Information:")
    print("=" * 50)

    # Check if we're in dev mode to show expected table names
    import os

    db_mode = os.getenv("DB_MODE", "production")
    table_prefix = os.getenv("DB_TABLE_PREFIX", "")

    if table_prefix:
        prefix = table_prefix
    elif db_mode == "dev":
        prefix = "dev_"
    else:
        prefix = ""

    print(f"Database Mode: {db_mode}")
    print(f"Table Prefix: '{prefix}'")
    print("\nGenerated documents will be stored in:")
    print(f"  📄 {prefix}stored_documents (main document metadata)")
    print(f"  📁 {prefix}document_files (file references)")
    print(f"  ❓ {prefix}generated_questions (if questions generated)")
    print(f"  🎯 {prefix}generation_sessions (generation session data)")

    print("\n💡 To view in Supabase:")
    print("  1. Go to your Supabase dashboard")
    print("  2. Navigate to Table Editor")
    print(f"  3. Look for tables with prefix '{prefix}'")
    print("  4. Check the stored_documents table for new entries")


def main():
    """Run test cases 1 and 2"""
    print("🚀 Running Test Cases 1 & 2 for Supabase Document Generation")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")

    # Show table information first
    check_supabase_tables()

    # Run the tests
    result1 = test_case_1_algebra_worksheet()
    result2 = test_case_2_geometry_notes()

    # Summary
    print("\n📊 Test Summary:")
    print("=" * 30)

    success_count = sum(1 for r in [result1, result2] if r is not None)
    print(f"Tests passed: {success_count}/2")

    if result1:
        print(f"✅ Case 1 - Storage ID: {result1.get('document_id', 'Not saved')}")
    if result2:
        print(f"✅ Case 2 - Storage ID: {result2.get('document_id', 'Not saved')}")

    print(f"\n🏁 Tests completed at: {datetime.now().isoformat()}")

    if success_count > 0:
        print("\n💾 Check your Supabase dashboard to see the generated documents!")
        print("   The documents should appear in the stored_documents table")


if __name__ == "__main__":
    main()
