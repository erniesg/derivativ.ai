"""
Integration tests for OpenAI streaming with live API calls.
Tests real OpenAI API connectivity and streaming functionality.
"""

import os
import time

import pytest
from dotenv import load_dotenv

from src.core.config import load_config
from src.models.llm_models import LLMRequest
from src.models.streaming_models import StreamingChunkType
from src.services.openai import OpenAILLMService

# Load environment variables
load_dotenv()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openai_streaming_integration():
    print("🌊 Testing OpenAI Streaming")
    print("=" * 40)

    try:
        # Load config and create service
        config = load_config()
        service = OpenAILLMService.from_config(config.llm_providers.openai)

        # Create streaming request
        request = LLMRequest(
            model="gpt-4o-mini",
            prompt="Write a haiku about artificial intelligence. Be creative!",
            max_tokens=100,
            temperature=0.8,
            stream=True,  # Enable streaming
        )

        print("📡 Starting streaming request...")
        start_time = time.time()

        content = ""
        chunk_count = 0
        first_chunk_time = None
        content_chunks = []
        done_chunks = []

        # Stream the response
        async for chunk in service.generate_stream(request):
            chunk_count += 1

            if chunk.chunk_type == StreamingChunkType.CONTENT:
                content_chunks.append(chunk)
                if not first_chunk_time:
                    first_chunk_time = time.time()
                    print(f"⚡ First chunk in {int((first_chunk_time - start_time) * 1000)}ms")

                # Print the delta in real-time
                if chunk.delta:
                    print(chunk.delta, end="", flush=True)
                    content += chunk.delta

            elif chunk.chunk_type == StreamingChunkType.DONE:
                done_chunks.append(chunk)
                print("\n\n✅ Stream completed!")
                metadata = chunk.metadata.get("final_response", {})
                print(f"   Total chunks: {chunk_count}")
                print(f"   Total tokens: {metadata.get('total_tokens', 0)}")
                print(f"   Cost: ${metadata.get('cost_estimate', 0):.6f}")
                print(f"   Total time: {int((time.time() - start_time) * 1000)}ms")

            elif chunk.chunk_type == StreamingChunkType.ERROR:
                print(f"\n❌ Error: {chunk.metadata.get('error', 'Unknown error')}")
                pytest.fail(f"Streaming error: {chunk.metadata.get('error', 'Unknown error')}")

        # Add assertions for proper pytest testing
        assert content_chunks, "Should receive content chunks"
        assert done_chunks, "Should receive completion signal"
        assert content, "Should accumulate content from chunks"

    except Exception as e:
        pytest.fail(f"Streaming test failed: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openai_non_streaming_integration():
    print("\n🔄 Testing OpenAI Non-Streaming")
    print("=" * 40)

    try:
        # Load config and create service
        config = load_config()
        service = OpenAILLMService.from_config(config.llm_providers.openai)

        # Create non-streaming request
        request = LLMRequest(
            model="gpt-4o-mini",
            prompt="Write a haiku about artificial intelligence. Be creative!",
            max_tokens=100,
            temperature=0.8,
            stream=False,  # Disable streaming
        )

        print("📡 Making non-streaming request...")
        start_time = time.time()

        # Get complete response
        response = await service.generate_non_stream(request)
        end_time = time.time()

        print("✅ Complete response received!")
        print(f"   Content: {response.content}")
        print(f"   Tokens: {response.tokens_used}")
        print(f"   Cost: ${response.cost_estimate:.6f}")
        print(f"   Latency: {int((end_time - start_time) * 1000)}ms")

        # Add assertions for proper pytest testing
        assert response.content, "Should receive content"
        assert response.tokens_used > 0, "Should track tokens"
        assert response.cost_estimate >= 0, "Should estimate cost"
        assert response.latency_ms > 0, "Should track latency"

    except Exception as e:
        pytest.fail(f"Non-streaming test failed: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_streaming_vs_non_streaming_performance():
    """Compare streaming vs non-streaming response times."""
    print("\n⚖️  Streaming vs Non-Streaming Comparison")
    print("=" * 50)

    prompt = "Explain quantum computing in exactly 50 words."

    # Load config and service
    config = load_config()
    service = OpenAILLMService.from_config(config.llm_providers.openai)

    # Test streaming time-to-first-chunk
    request_stream = LLMRequest(
        model="gpt-4o-mini", prompt=prompt, max_tokens=80, temperature=0.7, stream=True
    )

    stream_start = time.time()
    first_chunk_time = None

    async for chunk in service.generate_stream(request_stream):
        if chunk.chunk_type == StreamingChunkType.CONTENT and not first_chunk_time:
            first_chunk_time = time.time()
            break

    # Test non-streaming total time
    request_non_stream = LLMRequest(
        model="gpt-4o-mini", prompt=prompt, max_tokens=80, temperature=0.7, stream=False
    )

    non_stream_start = time.time()
    response = await service.generate_non_stream(request_non_stream)
    non_stream_end = time.time()

    # Assertions for performance comparison
    streaming_ttfc = (first_chunk_time - stream_start) * 1000 if first_chunk_time else None
    non_streaming_total = (non_stream_end - non_stream_start) * 1000

    print(f"   Streaming time-to-first-chunk: {int(streaming_ttfc)}ms")
    print(f"   Non-streaming total time: {int(non_streaming_total)}ms")

    # Performance assertions
    assert streaming_ttfc is not None, "Should get first chunk timing"
    assert streaming_ttfc > 0, "Time-to-first-chunk should be positive"
    assert non_streaming_total > 0, "Non-streaming time should be positive"
    assert streaming_ttfc < 30000, "Time-to-first-chunk should be under 30 seconds"
    assert non_streaming_total < 30000, "Non-streaming should complete under 30 seconds"
