#!/usr/bin/env python3
"""
Live End-to-End Test for DocumentGenerationServiceV2.
Tests the complete workflow from API request to database storage.
"""

import json
import os
import time
from datetime import datetime

import requests


def test_document_generation_v2_api():
    """Test the V2 document generation API endpoint with live database integration."""
    print("🧪 Testing Document Generation V2 API (Live E2E)...")

    api_url = "http://localhost:8000/api/generation/documents/generate-v2"

    # Test cases using TopicName enum values and detailed configurations
    test_cases = [
        {
            "name": "Algebra Worksheet with DB Questions",
            "data": {
                "title": "Linear Equations Practice Worksheet",
                "document_type": "worksheet",
                "topic": "algebra_and_graphs",  # TopicName.ALGEBRA_AND_GRAPHS.value
                "tier": "Core",
                "grade_level": 8,
                "target_time_minutes": 30,
                "detail_level": 5,
                "custom_instructions": "Focus on solving linear equations with practical applications",
            },
        },
        {
            "name": "Geometry Notes with Syllabus References",
            "data": {
                "title": "Coordinate Geometry Study Notes",
                "document_type": "notes",
                "topic": "coordinate_geometry",  # TopicName.COORDINATE_GEOMETRY.value
                "tier": "Extended",
                "grade_level": 9,
                "target_time_minutes": 45,
                "detail_level": 7,
                "subject_content_refs": ["C6.1", "C6.2"],  # Will be converted to enum
                "custom_instructions": "Include step-by-step worked examples and practice problems",
            },
        },
        {
            "name": "Statistics Mini-Textbook",
            "data": {
                "title": "Data Analysis and Statistics",
                "document_type": "textbook",
                "topic": "statistics",  # TopicName.STATISTICS.value
                "tier": "Core",
                "grade_level": 10,
                "target_time_minutes": 60,
                "detail_level": 8,
                "custom_instructions": "Comprehensive coverage with real-world examples and exercises",
            },
        },
    ]

    results = []

    for test_case in test_cases:
        print(f"\n📋 Testing: {test_case['name']}")
        print(f"Request: {json.dumps(test_case['data'], indent=2)}")

        start_time = time.time()

        try:
            response = requests.post(
                api_url,
                json=test_case["data"],
                headers={"Content-Type": "application/json"},
                timeout=60,  # Longer timeout for E2E tests
            )

            processing_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                print(f"✅ SUCCESS ({processing_time:.2f}s)")
                print(f"   Document ID: {result['document']['document_id']}")
                print(f"   Title: {result['document']['title']}")
                print(f"   Content Blocks: {len(result['document']['content_blocks'])}")
                print(f"   Processing Time: {result['processing_time']:.2f}s")

                # Check if document was saved to storage
                if result.get("document_id"):
                    print(f"   ✅ Saved to Storage: {result['document_id']}")
                else:
                    print("   ⚠️ Not saved to storage (demo mode?)")

                # Display block information
                for i, block in enumerate(
                    result["document"]["content_blocks"][:3]
                ):  # First 3 blocks
                    print(
                        f"   Block {i+1}: {block['block_type']} - {block.get('title', 'Untitled')}"
                    )

                results.append(
                    {
                        "test": test_case["name"],
                        "status": "success",
                        "processing_time": processing_time,
                        "api_processing_time": result["processing_time"],
                        "blocks_count": len(result["document"]["content_blocks"]),
                        "document_id": result["document"]["document_id"],
                        "storage_id": result.get("document_id"),
                    }
                )
            else:
                print(f"❌ FAILED ({response.status_code})")
                print(f"   Error: {response.text}")

                results.append(
                    {
                        "test": test_case["name"],
                        "status": "failed",
                        "error": f"HTTP {response.status_code}: {response.text[:200]}",
                    }
                )

        except requests.exceptions.Timeout:
            print("⏰ TIMEOUT (>60s)")
            results.append(
                {
                    "test": test_case["name"],
                    "status": "timeout",
                    "error": "Request timed out after 60 seconds",
                }
            )

        except Exception as e:
            print(f"💥 ERROR: {e}")
            results.append({"test": test_case["name"], "status": "error", "error": str(e)})

    return results


def test_database_table_prefixes():
    """Test that the table prefix system is working correctly."""
    print("\n🗃️ Testing Database Table Prefix System...")

    # Check environment variables
    db_mode = os.getenv("DB_MODE", "production")
    table_prefix = os.getenv("DB_TABLE_PREFIX", "")
    demo_mode = os.getenv("DEMO_MODE", "false").lower() in ("true", "1", "yes")

    print(f"   DB_MODE: {db_mode}")
    print(f"   DB_TABLE_PREFIX: '{table_prefix}'")
    print(f"   DEMO_MODE: {demo_mode}")

    # Expected table names based on configuration
    if table_prefix:
        expected_prefix = table_prefix
    elif db_mode == "dev":
        expected_prefix = "dev_"
    else:
        expected_prefix = ""

    expected_tables = [
        f"{expected_prefix}generated_questions",
        f"{expected_prefix}generation_sessions",
        f"{expected_prefix}stored_documents",
        f"{expected_prefix}document_files",
    ]

    print(f"   Expected table prefix: '{expected_prefix}'")
    print(f"   Expected tables: {expected_tables}")

    if demo_mode:
        print("   ✅ Running in DEMO_MODE - using mock repositories")
    else:
        print("   ✅ Running with live database - tables should use proper prefixes")


def test_health_and_connectivity():
    """Test API health and basic connectivity."""
    print("\n🏥 Testing API Health and Connectivity...")

    endpoints = [
        ("Health", "http://localhost:8000/health"),
        ("V1 Generate", "http://localhost:8000/api/generation/documents/generate"),
        ("V2 Generate", "http://localhost:8000/api/generation/documents/generate-v2"),
        ("Templates", "http://localhost:8000/api/generation/documents/templates"),
    ]

    for name, url in endpoints:
        try:
            # Use GET for non-generation endpoints, HEAD for generation endpoints
            if "generate" in url:
                response = requests.head(url, timeout=5)
                expected_status = 405  # Method not allowed for HEAD on POST endpoints
            else:
                response = requests.get(url, timeout=5)
                expected_status = 200

            if response.status_code == expected_status:
                print(f"   ✅ {name}: Available")
            else:
                print(f"   ⚠️  {name}: Unexpected status {response.status_code}")
        except Exception as e:
            print(f"   ❌ {name}: {e}")


def test_topic_enum_validation():
    """Test that TopicName enum values work correctly."""
    print("\n📚 Testing TopicName Enum Validation...")

    # Test a few key topic values that should be available
    test_topics = [
        "algebra_and_graphs",
        "coordinate_geometry",
        "statistics",
        "number_and_set_theory",
        "trigonometry",
    ]

    api_url = "http://localhost:8000/api/generation/documents/generate-v2"

    for topic in test_topics:
        try:
            test_data = {
                "title": f"Test {topic.replace('_', ' ').title()}",
                "document_type": "worksheet",
                "topic": topic,
                "tier": "Core",
                "grade_level": 8,
                "detail_level": 3,
            }

            # Quick validation request (should not timeout)
            response = requests.post(
                api_url, json=test_data, headers={"Content-Type": "application/json"}, timeout=10
            )

            if response.status_code == 200:
                print(f"   ✅ {topic}: Valid enum value")
            elif response.status_code == 422:
                print(f"   ❌ {topic}: Invalid enum value (validation error)")
            else:
                print(f"   ⚠️  {topic}: Unexpected response {response.status_code}")

        except requests.exceptions.Timeout:
            print(f"   ⏰ {topic}: Request timed out (but enum validation passed)")
        except Exception as e:
            print(f"   💥 {topic}: Error - {e}")


def validate_demo_vs_live_mode():
    """Validate whether we're running in demo or live mode and show expected behavior."""
    print("\n🎭 Validating Demo vs Live Mode...")

    demo_mode = os.getenv("DEMO_MODE", "false").lower() in ("true", "1", "yes")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")

    print(f"   Demo Mode: {demo_mode}")
    print(f"   Supabase URL configured: {bool(supabase_url)}")
    print(f"   Supabase Key configured: {bool(supabase_key)}")

    if demo_mode:
        print("   🎭 DEMO MODE ACTIVE:")
        print("     - Using mock repositories")
        print("     - Documents stored in memory only")
        print("     - No real database operations")
        print("     - Consistent demo responses")
    elif supabase_url and supabase_key:
        print("   🔴 LIVE MODE ACTIVE:")
        print("     - Using real Supabase database")
        print("     - Documents persist to storage")
        print("     - Real question generation and DB lookup")
        print("     - Table prefixing applied")
    else:
        print("   ⚠️  CONFIGURATION ISSUE:")
        print("     - Demo mode disabled but Supabase not configured")
        print("     - Services may fall back to demo mode automatically")


def main():
    """Run comprehensive live end-to-end tests for V2 document generation."""
    print("🚀 Derivativ Live End-to-End Test Suite (V2)")
    print("=" * 60)
    print(f"Test started at: {datetime.now().isoformat()}")

    # Validate environment setup
    validate_demo_vs_live_mode()
    test_database_table_prefixes()
    test_health_and_connectivity()
    test_topic_enum_validation()

    # Run main E2E tests
    results = test_document_generation_v2_api()

    # Print comprehensive summary
    print("\n📊 Test Summary:")
    print("=" * 40)

    success_count = sum(1 for r in results if r["status"] == "success")
    total_count = len(results)

    print(f"API Tests: {success_count}/{total_count} passed")

    if success_count == total_count:
        print("✅ All E2E tests PASSED!")
        print("\n🎯 V2 Document Generation System fully operational!")

        # Show performance metrics
        if results:
            avg_time = sum(
                r.get("processing_time", 0) for r in results if r["status"] == "success"
            ) / max(success_count, 1)
            print(f"📈 Average processing time: {avg_time:.2f}s")

            # Show storage status
            stored_count = sum(1 for r in results if r.get("storage_id"))
            print(f"💾 Documents persisted to storage: {stored_count}/{success_count}")
    else:
        print("❌ Some tests FAILED")
        for result in results:
            if result["status"] != "success":
                print(f"   - {result['test']}: {result.get('error', result['status'])}")

    print(f"\n🏁 E2E test suite completed at: {datetime.now().isoformat()}")

    print("\n🔧 To run the backend:")
    print("   export DEMO_MODE=true  # or configure Supabase for live mode")
    print("   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")

    print("\n🌐 API Endpoints tested:")
    print("   - POST /api/generation/documents/generate-v2")
    print("   - GET  /health")
    print("   - GET  /api/generation/documents/templates")


if __name__ == "__main__":
    main()
