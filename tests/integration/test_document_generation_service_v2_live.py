#!/usr/bin/env python3
"""
Integration test for document generation v2 with blocks architecture.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.models.document_generation_v2 import DocumentGenerationRequestV2
from src.models.document_models import DocumentType
from src.models.enums import Tier, TopicName
from src.services.document_generation_service_v2 import DocumentGenerationServiceV2
from src.services.llm_factory import LLMFactory
from src.services.prompt_manager import PromptManager


async def test_document_generation_v2():
    """Test the new document generation service end-to-end."""

    # Mock LLM factory and prompt manager
    llm_factory = MagicMock(spec=LLMFactory)
    prompt_manager = MagicMock(spec=PromptManager)

    # Mock LLM service
    mock_llm_service = AsyncMock()
    llm_factory.get_service.return_value = mock_llm_service

    # Mock template rendering - let's see what the real template would look like
    def mock_render_prompt(config, **kwargs):
        return f"""
        Generated template for {config.variables.get('topic', 'unknown topic')}

        Syllabus refs: {config.variables.get('syllabus_refs', [])}
        Detailed content: {list(config.variables.get('detailed_syllabus_content', {}).keys())}

        Detail level: {config.variables.get('detail_level', 5)}
        Selected blocks: {[b['block_type'] for b in config.variables.get('selected_blocks', [])]}
        """

    prompt_manager.render_prompt = AsyncMock(side_effect=mock_render_prompt)

    # Mock LLM response with valid JSON
    mock_response = MagicMock()
    mock_response.content = """
    {
        "enhanced_title": "Algebra Practice Worksheet",
        "introduction": "This worksheet covers key algebraic concepts.",
        "blocks": [
            {
                "block_type": "learning_objectives",
                "content": {
                    "objectives": [
                        "Solve linear equations",
                        "Factorize quadratic expressions"
                    ]
                },
                "estimated_minutes": 2,
                "reasoning": "Essential learning goals"
            },
            {
                "block_type": "practice_questions",
                "content": {
                    "questions": [
                        {
                            "text": "Solve 2x + 5 = 13",
                            "marks": 2,
                            "difficulty": "medium",
                            "answer": "x = 4"
                        },
                        {
                            "text": "Factorize x² + 5x + 6",
                            "marks": 3,
                            "difficulty": "medium",
                            "answer": "(x + 2)(x + 3)"
                        }
                    ],
                    "include_answers": true
                },
                "estimated_minutes": 6,
                "reasoning": "Practice essential skills"
            }
        ],
        "total_estimated_minutes": 8,
        "actual_detail_level": 5,
        "generation_reasoning": "Selected blocks for focused practice",
        "coverage_notes": "Covers basic algebra concepts",
        "personalization_applied": []
    }
    """
    mock_llm_service.generate_non_stream.return_value = mock_response

    # Create service (without question generation service for now)
    service = DocumentGenerationServiceV2(
        llm_factory=llm_factory, prompt_manager=prompt_manager, question_generation_service=None
    )

    # Create request with proper enums
    request = DocumentGenerationRequestV2(
        document_type=DocumentType.WORKSHEET,
        title="Algebra Practice",
        topic=TopicName.ALGEBRA_AND_GRAPHS,
        tier=Tier.CORE,
        target_duration_minutes=15,
        include_questions=True,
        num_questions=2,
    )

    print("🚀 Testing Document Generation V2")
    print(f"Topic: {request.topic}")
    print(f"Syllabus refs: {request.get_syllabus_refs()}")
    print(f"Detail level: {request.get_effective_detail_level()}")

    # Generate document
    result = await service.generate_document(request)

    if result.success:
        print("\n✅ Document generation successful!")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Document title: {result.document.title}")
        print(f"Blocks generated: {len(result.document.content_structure.blocks)}")
        print(f"Total estimated time: {result.document.total_estimated_minutes} minutes")

        # Show rendered content
        markdown_content = result.document.get_markdown_content()
        print("\n📄 Generated content preview (first 500 chars):")
        print(markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content)

        return True
    else:
        print(f"\n❌ Document generation failed: {result.error_message}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_document_generation_v2())
    if success:
        print("\n🎉 All tests passed! Document generation V2 is working.")
    else:
        print("\n💥 Tests failed. Check the implementation.")
