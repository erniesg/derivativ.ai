#!/usr/bin/env python3
"""
Full-stack integration test script.
Tests the complete document generation workflow from frontend to backend.
"""

import json
import time

import requests


def test_document_generation_api():
    """Test the document generation API endpoint."""
    print("🧪 Testing Document Generation API...")

    api_url = "http://localhost:8000/api/documents/generate"

    # Test data matching the frontend TeacherDashboard
    test_cases = [
        {
            "name": "Worksheet Generation",
            "data": {
                "document_type": "worksheet",
                "detail_level": "medium",
                "title": "Algebra Practice Worksheet",
                "topic": "linear_equations",
                "tier": "Core",
                "grade_level": 7,
                "auto_include_questions": True,
                "max_questions": 5,
                "include_answers": True,
                "include_working": True
            }
        },
        {
            "name": "Notes Generation",
            "data": {
                "document_type": "notes",
                "detail_level": "comprehensive",
                "title": "Geometry Study Notes",
                "topic": "geometry",
                "tier": "Core",
                "grade_level": 8,
                "auto_include_questions": True,
                "max_questions": 3,
                "custom_instructions": "Include visual diagrams and step-by-step explanations"
            }
        }
    ]

    results = []

    for test_case in test_cases:
        print(f"\n📋 Testing: {test_case['name']}")
        print(f"Request: {json.dumps(test_case['data'], indent=2)}")

        start_time = time.time()

        try:
            response = requests.post(
                api_url,
                json=test_case['data'],
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            processing_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                print(f"✅ SUCCESS ({processing_time:.2f}s)")
                print(f"   Document ID: {result['document']['document_id']}")
                print(f"   Title: {result['document']['title']}")
                print(f"   Sections: {result['sections_generated']}")
                print(f"   Processing Time: {result['processing_time']:.2f}s")

                results.append({
                    "test": test_case['name'],
                    "status": "success",
                    "processing_time": processing_time,
                    "api_processing_time": result['processing_time'],
                    "sections": result['sections_generated']
                })
            else:
                print(f"❌ FAILED ({response.status_code})")
                print(f"   Error: {response.text}")

                results.append({
                    "test": test_case['name'],
                    "status": "failed",
                    "error": f"HTTP {response.status_code}: {response.text}"
                })

        except requests.exceptions.Timeout:
            print("⏰ TIMEOUT (>30s)")
            results.append({
                "test": test_case['name'],
                "status": "timeout",
                "error": "Request timed out after 30 seconds"
            })

        except Exception as e:
            print(f"💥 ERROR: {e}")
            results.append({
                "test": test_case['name'],
                "status": "error",
                "error": str(e)
            })

    return results


def test_template_endpoints():
    """Test the template management endpoints."""
    print("\n🎨 Testing Template Management...")

    base_url = "http://localhost:8000/api/documents"

    try:
        # Test get templates
        response = requests.get(f"{base_url}/templates")
        if response.status_code == 200:
            templates = response.json()
            print(f"✅ Templates retrieved: {list(templates.keys())}")
            return True
        else:
            print(f"❌ Failed to get templates: {response.status_code}")
            return False

    except Exception as e:
        print(f"💥 Template test error: {e}")
        return False


def test_health_endpoints():
    """Test API health and connectivity."""
    print("\n🏥 Testing Health Endpoints...")

    endpoints = [
        ("Root", "http://localhost:8000/"),
        ("Health", "http://localhost:8000/health"),
        ("Docs", "http://localhost:8000/docs")
    ]

    for name, url in endpoints:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code in [200, 404]:  # 404 is OK for some endpoints
                print(f"✅ {name}: Available")
            else:
                print(f"⚠️  {name}: {response.status_code}")
        except Exception as e:
            print(f"❌ {name}: {e}")


def print_frontend_instructions():
    """Print instructions for testing the frontend."""
    print("\n🌐 Frontend Testing Instructions:")
    print("=" * 50)
    print("1. Navigate to the frontend directory:")
    print("   cd /Users/erniesg/code/erniesg/derivativ")
    print("\n2. Start the frontend development server:")
    print("   npm run dev")
    print("\n3. Open your browser to: http://localhost:5173")
    print("\n4. Navigate to the Teacher Dashboard")
    print("\n5. Generate a document with these settings:")
    print("   - Material Type: worksheet")
    print("   - Topics: Select 'Algebra'")
    print("   - Detail Level: 5/10")
    print("   - Target Level: IGCSE")
    print("\n6. Click 'Generate Material' and verify:")
    print("   - Success alert appears")
    print("   - Processing time is displayed")
    print("   - Console shows API response")
    print("\n7. Check the browser console for detailed logs")


def main():
    """Run full-stack integration tests."""
    print("🚀 Derivativ Full-Stack Integration Test")
    print("=" * 50)

    # Test API endpoints
    results = test_document_generation_api()
    test_template_endpoints()
    test_health_endpoints()

    # Print summary
    print("\n📊 Test Summary:")
    print("=" * 30)

    success_count = sum(1 for r in results if r['status'] == 'success')
    total_count = len(results)

    print(f"API Tests: {success_count}/{total_count} passed")

    if success_count == total_count:
        print("✅ All API tests PASSED!")
        print("\n🎯 Backend is ready for frontend integration!")
    else:
        print("❌ Some tests FAILED")
        for result in results:
            if result['status'] != 'success':
                print(f"   - {result['test']}: {result.get('error', result['status'])}")

    # Print frontend instructions
    print_frontend_instructions()

    print("\n🏁 Integration test complete!")
    print("\nNote: Make sure to start the backend with DEMO_MODE=true:")
    print("export DEMO_MODE=true && uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")


if __name__ == "__main__":
    main()
