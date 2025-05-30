#!/usr/bin/env python3
"""
Test script for Claude 4 integration via Amazon Bedrock
Tests Amazon Bedrock ServerModel integration for Claude 4 Sonnet and Opus models.
"""

import os
import sys
import asyncio

# Add project root to Python path for clean imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from smolagents import AmazonBedrockServerModel
from dotenv import load_dotenv

load_dotenv()


async def test_claude4_sonnet_bedrock():
    """Test Claude 4 Sonnet via Amazon Bedrock"""
    print("🧪 Testing Claude 4 Sonnet via Amazon Bedrock...")

    # Check AWS credentials
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "us-east-1")

    if not aws_access_key or not aws_secret_key:
        print("❌ AWS credentials not found in environment variables")
        print("   Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return False

    print(f"✅ AWS Region: {aws_region}")
    print(f"✅ AWS Access Key: {aws_access_key[:8]}...")

    try:
        # Create Claude 4 Sonnet model
        model = AmazonBedrockServerModel(
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            client_kwargs={'region_name': aws_region}
        )

        print("✅ Claude 4 Sonnet model created successfully")

        # Test a simple math question generation prompt
        test_prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Generate a simple IGCSE Grade 5 algebra question in this format: Question: [question text]. Answer: [answer]. Marks: [number]"
                    }
                ]
            }
        ]

        print("🔄 Testing model response...")
        response = model(test_prompt)

        print(f"✅ Model response received:")
        print(f"   Type: {type(response)}")
        if hasattr(response, 'content'):
            print(f"   Content: {response.content[:300]}...")
        else:
            print(f"   Response: {str(response)[:300]}...")

        return True

    except Exception as e:
        print(f"❌ Error testing Claude 4 Sonnet: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False


async def test_claude4_opus_bedrock():
    """Test Claude 4 Opus via Amazon Bedrock"""
    print("\n🧪 Testing Claude 4 Opus via Amazon Bedrock...")

    # Check AWS credentials
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "us-east-1")

    if not aws_access_key or not aws_secret_key:
        print("❌ AWS credentials not found")
        return False

    try:
        # Create Claude 4 Opus model
        model = AmazonBedrockServerModel(
            model_id="us.anthropic.claude-opus-4-20250514-v1:0",
            client_kwargs={'region_name': aws_region}
        )

        print("✅ Claude 4 Opus model created successfully")

        # Test a more complex question generation prompt
        test_prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Generate a challenging IGCSE Grade 8 geometry question involving circles and angles. Include a complete marking scheme."
                    }
                ]
            }
        ]

        print("🔄 Testing model response...")
        response = model(test_prompt)

        print(f"✅ Model response received:")
        print(f"   Type: {type(response)}")
        if hasattr(response, 'content'):
            print(f"   Content: {response.content[:300]}...")
        else:
            print(f"   Response: {str(response)[:300]}...")

        return True

    except Exception as e:
        print(f"❌ Error testing Claude 4 Opus: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False


async def test_bedrock_integration():
    """Run comprehensive Bedrock integration tests"""
    print("🔄 Starting Amazon Bedrock Claude 4 integration tests...\n")

    # Test Claude 4 Sonnet
    sonnet_success = await test_claude4_sonnet_bedrock()

    # Test Claude 4 Opus
    opus_success = await test_claude4_opus_bedrock()

    print(f"\n📊 Test Results:")
    print(f"   Claude 4 Sonnet: {'✅ PASS' if sonnet_success else '❌ FAIL'}")
    print(f"   Claude 4 Opus: {'✅ PASS' if opus_success else '❌ FAIL'}")

    if sonnet_success or opus_success:
        print("\n🎉 At least one Claude 4 model is working!")
        print("   Your Bedrock integration is properly configured.")
        return True
    else:
        print("\n💡 To use Claude 4 models:")
        print("   1. Set up AWS credentials with Bedrock access")
        print("   2. Request access to Claude 4 models in your AWS region")
        print("   3. Ensure you're in a supported region (us-east-1, us-east-2, us-west-2)")
        print("   4. Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION in .env")
        return False


if __name__ == "__main__":
    asyncio.run(test_bedrock_integration())
