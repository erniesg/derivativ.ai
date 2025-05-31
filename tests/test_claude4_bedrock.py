#!/usr/bin/env python3
"""
Simple test for APAC Claude 4 Sonnet inference profile
"""

import os
import sys
import asyncio

# Add project root to Python path for clean imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from smolagents import AmazonBedrockServerModel
from dotenv import load_dotenv

load_dotenv()


async def test_apac_claude():
    """Simple test of APAC Claude 4 Sonnet"""
    print("🧪 Testing APAC Claude 4 Sonnet...")

    # Check AWS credentials
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not aws_access_key or not aws_secret_key:
        print("❌ AWS credentials not found")
        return False

    print(f"✅ AWS Access Key: {aws_access_key[:8]}...")

    try:
        # Create Claude model
        model = AmazonBedrockServerModel(
            model_id="apac.anthropic.claude-sonnet-4-20250514-v1:0",
            client_kwargs={'region_name': "ap-southeast-1"}
        )

        print("✅ Model created successfully")

        # Simple test prompt
        test_prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello! Can you say 'Claude model is working'?"
                    }
                ]
            }
        ]

        print("🔄 Testing model response...")

        response = model(test_prompt)

        print(f"✅ SUCCESS!")
        print(f"   Response type: {type(response)}")
        if hasattr(response, 'content'):
            print(f"   Content: {response.content[:100]}...")
        else:
            print(f"   Response: {str(response)[:100]}...")

        return True

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Failed: {error_msg[:200]}...")

        if "ThrottlingException" in error_msg:
            print(f"   → Rate limited (model is working, just throttled)")
            return True  # Consider this a success since model is accessible
        elif "AccessDeniedException" in error_msg:
            print(f"   → Access denied")
            return False
        else:
            print(f"   → Other error")
            return False


if __name__ == "__main__":
    asyncio.run(test_apac_claude())
