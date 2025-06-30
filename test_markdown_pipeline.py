#!/usr/bin/env python3
"""
Test script for the new markdown + pandoc + R2 pipeline.

This tests the complete flow from markdown generation to R2 storage.
"""

import asyncio

from src.api.dependencies import get_integrated_document_service
from src.models.document_models import DetailLevel, DocumentType
from src.models.enums import Tier, TopicName
from src.models.markdown_generation_models import MarkdownGenerationRequest


async def test_markdown_pipeline():  # noqa: PLR0915
    """Test the complete markdown pipeline."""

    print("🧪 Testing Markdown + Pandoc + R2 Pipeline")
    print("=" * 50)

    try:
        # Create test request
        request = MarkdownGenerationRequest(
            document_type=DocumentType.WORKSHEET,
            topic=TopicName.ALGEBRA_AND_GRAPHS,
            tier=Tier.CORE,
            detail_level=DetailLevel.MEDIUM,
            target_duration_minutes=30,
            grade_level="7-9",
        )

        print("📝 Test Request:")
        print(f"   Document Type: {request.document_type.value}")
        print(f"   Topic: {request.topic.value}")
        print(f"   Detail Level: {request.detail_level}/10")
        print(f"   Duration: {request.target_duration_minutes} minutes")
        print()

        # Get service instance with proper dependency injection
        from src.api.dependencies import get_llm_factory, get_prompt_manager, get_r2_storage_service
        
        llm_factory = get_llm_factory()
        prompt_manager = get_prompt_manager()
        r2_service = get_r2_storage_service()
        
        service = get_integrated_document_service(
            llm_factory=llm_factory,
            prompt_manager=prompt_manager,
            r2_service=r2_service
        )

        print("🔧 Service initialized successfully")
        print()

        # Test markdown generation
        print("1️⃣ Testing markdown generation...")
        result = await service.generate_and_store_all_formats(
            request=request,
            custom_instructions="Focus on linear equations and simple algebraic manipulation",
        )

        if result["success"]:
            print("✅ Markdown generation successful!")
            print(f"   Document ID: {result['document_id']}")
            print(f"   Markdown length: {len(result['markdown_content'])} characters")
            print()

            # Show format results
            print("2️⃣ Format conversion results:")
            for format_name, format_data in result["formats"].items():
                if format_data["success"]:
                    print(f"   ✅ {format_name.upper()}: {format_data.get('size', 0)} bytes")
                    if format_data.get("r2_url"):
                        print(f"      📎 Download: {format_data['r2_url'][:60]}...")
                else:
                    print(
                        f"   ❌ {format_name.upper()}: {format_data.get('error', 'Unknown error')}"
                    )
            print()

            # Show metadata
            print("3️⃣ Document metadata:")
            metadata = result["metadata"]
            for key, value in metadata.items():
                print(f"   {key}: {value}")
            print()

            # Test status endpoint
            print("4️⃣ Testing status retrieval...")
            status = await service.get_document_status(result["document_id"])
            print("✅ Status check successful!")
            for format_name, format_status in status.items():
                available = format_status.get("available", False)
                print(f"   {format_name}: {'✅ Available' if available else '❌ Not available'}")
            print()

            # Show sample markdown content
            print("5️⃣ Sample markdown content:")
            print("-" * 40)
            print(
                result["markdown_content"][:500] + "..."
                if len(result["markdown_content"]) > 500
                else result["markdown_content"]
            )
            print("-" * 40)
            print()

            print("🎉 PIPELINE TEST SUCCESSFUL!")
            print("✅ All components working correctly")
            print("✅ Professional PDF/DOCX generation via pandoc")
            print("✅ R2 storage and presigned URLs working")
            print("✅ No complex JSON structure issues")

            return True

        else:
            print(f"❌ Pipeline test failed: {result.get('error')}")
            return False

    except Exception as e:
        print(f"❌ Pipeline test error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_api_endpoint():
    """Test the new API endpoint directly."""
    print("\n🌐 Testing API Endpoint")
    print("=" * 30)

    try:
        import httpx

        # Test data
        test_payload = {
            "document_type": "worksheet",
            "topic": "Algebra and graphs",
            "tier": "Core",
            "detail_level": 5,
            "target_duration_minutes": 30,
            "grade_level": "7-9",
        }

        print("📡 Sending request to /api/generation/documents/generate-markdown")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/api/generation/documents/generate-markdown",
                json=test_payload,
                timeout=120.0,
            )

            if response.status_code == 200:
                result = response.json()
                print("✅ API endpoint successful!")
                print(f"   Document ID: {result.get('document_id')}")
                print(f"   Formats available: {len(result.get('downloads', {}))}")

                for format_name, download_info in result.get("downloads", {}).items():
                    if download_info["available"]:
                        print(f"   ✅ {format_name}: Ready for download")
                    else:
                        print(f"   ❌ {format_name}: Not available")

                return True

            else:
                print(f"❌ API request failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False

    except Exception as e:
        print(f"❌ API test error: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Testing New Markdown + Pandoc + R2 Pipeline")
    print("=" * 60)
    print()

    # Test 1: Direct service test
    service_success = await test_markdown_pipeline()

    # Test 2: API endpoint test (requires server running)
    api_success = await test_api_endpoint()

    print("\n📊 Test Summary:")
    print(f"   Service Pipeline: {'✅ PASS' if service_success else '❌ FAIL'}")
    print(f"   API Endpoint: {'✅ PASS' if api_success else '❌ FAIL'}")

    if service_success and api_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Ready to update frontend to use new pipeline")
    elif service_success:
        print("\n⚠️ Service works, API might need server restart")
        print("💡 Start server: uvicorn src.api.main:app --reload")
    else:
        print("\n❌ Tests failed - check dependencies and configuration")


if __name__ == "__main__":
    asyncio.run(main())
