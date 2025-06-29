#!/usr/bin/env python3
"""Debug script to test storage saving in V2 document generation."""

import json
import os

import requests

# Set demo mode
os.environ["DEMO_MODE"] = "true"


def test_single_generation():
    """Test a single document generation to debug storage."""
    url = "http://localhost:8000/api/generation/documents/generate-v2"

    data = {
        "title": "Debug Test Worksheet",
        "document_type": "worksheet",
        "topic": "Number",
        "tier": "Core",
        "grade_level": 8,
        "detail_level": 3,
    }

    print("🧪 Testing single document generation for storage debugging...")
    print(f"Request: {json.dumps(data, indent=2)}")

    try:
        response = requests.post(
            url, json=data, headers={"Content-Type": "application/json"}, timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ Generation successful!")
            print(f"Document ID: {result['document']['document_id']}")

            # Check generation insights
            insights = result.get("generation_insights", {})
            print(f"Generation insights: {json.dumps(insights, indent=2)}")

            # Look for storage information
            storage_id = insights.get("document_id")
            if storage_id:
                print(f"📁 Storage ID found: {storage_id}")
            else:
                print("⚠️ No storage ID in generation insights")

            # Check content blocks
            content_blocks = result["document"].get("content_structure", {}).get("blocks", [])
            print(f"Content blocks: {len(content_blocks)}")

            return True
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False

    except Exception as e:
        print(f"💥 Exception: {e}")
        return False


if __name__ == "__main__":
    test_single_generation()
