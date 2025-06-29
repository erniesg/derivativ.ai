"""
Service-specific content blocks for special operations.

These blocks handle specific service integrations like question generation
and answer key formatting.
"""

from typing import Any

from src.models.content_blocks import BlockRenderFormat, ContentBlock


class AnswerKeyBlock(ContentBlock):
    """A specialized block that formats answer keys from questions."""

    block_type: str = "answer_key"
    estimated_minutes: int = 5  # Time to review answer key

    async def render(self, content: dict[str, Any], format: BlockRenderFormat = BlockRenderFormat.MARKDOWN) -> str:
        """
        Render answer key from questions with marking schemes.

        Args:
            content: Dict containing 'questions' list of Question objects
            format: Output format for rendering

        Returns:
            Formatted answer key
        """
        questions = content.get("questions", [])

        if format == BlockRenderFormat.MARKDOWN:
            output = ["## Answer Key\n"]

            for i, q in enumerate(questions):
                output.append(f"### Question {i+1}")
                output.append(f"**Question:** {q.raw_text_content}\n")

                if q.solution_and_marking_scheme:
                    output.append("**Marking Scheme:**")
                    for criterion in q.solution_and_marking_scheme.mark_allocation_criteria:
                        output.append(f"- ({criterion.mark_code_display}) {criterion.criterion_text} - {criterion.marks_value} mark(s)")

                    if q.solution_and_marking_scheme.final_answers_summary:
                        final_answer = q.solution_and_marking_scheme.final_answers_summary[0].answer_text
                        output.append(f"\n**Final Answer:** {final_answer}")
                output.append("")

            return "\n".join(output)
        # TODO: Implement other formats
        return ""


class QuestionGeneratorBlock(ContentBlock):
    """
    A service integration block that generates questions dynamically.

    This block integrates with the QuestionGenerationService to create
    new questions based on specified parameters.
    """

    block_type: str = "question_generator"
    estimated_minutes: int = 0  # Dynamic based on generated questions

    def __init__(self, question_generation_service=None, **kwargs):
        super().__init__(**kwargs)
        self.question_generation_service = question_generation_service

    async def render(self, content: dict[str, Any], format: BlockRenderFormat = BlockRenderFormat.MARKDOWN) -> str:
        """
        Generate questions using the service.

        This is a special block that doesn't render directly but triggers
        question generation. The generated questions should be passed to
        other blocks for rendering.
        """
        if not self.question_generation_service:
            return "<!-- Question generation service not available -->"

        # Extract generation parameters
        request_params = content.get("generation_params", {})

        try:
            from src.models.question_models import GenerationRequest
            generation_request = GenerationRequest(**request_params)
            session = await self.question_generation_service.generate_questions(generation_request)

            # Store generated questions in content for other blocks to use
            content["generated_questions"] = session.questions

            return f"<!-- Generated {len(session.questions)} questions -->"
        except Exception as e:
            return f"<!-- Question generation failed: {e!s} -->"
