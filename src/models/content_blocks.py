"""
Content block models for document generation.

Provides reusable content components that can be composed into different
document types with varying levels of detail.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class BlockRenderFormat(str, Enum):
    """Supported rendering formats for content blocks."""
    MARKDOWN = "markdown"
    HTML = "html"
    LATEX = "latex"
    PLAIN_TEXT = "plain_text"


class ContentBlock(ABC, BaseModel):
    """
    Base class for all content blocks.

    Each block represents a reusable educational component that can be
    rendered in different formats and composed into documents.
    """
    block_type: str = Field(..., description="Type identifier for this block")
    estimated_minutes: int = Field(..., description="Base time estimate to complete/read this block")

    @abstractmethod
    async def render(self, content: dict[str, Any], format: BlockRenderFormat = BlockRenderFormat.MARKDOWN) -> str:
        """
        Render block content for specific output format.

        Args:
            content: Block-specific content data
            format: Output format for rendering

        Returns:
            Rendered content as string
        """
        pass

    class Config:
        """Allow arbitrary types for ABC compatibility."""
        arbitrary_types_allowed = True


class LearningObjectivesBlock(ContentBlock):
    """
    Learning objectives that students will achieve.

    Typically 2-5 bullet points that clearly state what students
    will know or be able to do after completing the material.
    """
    block_type: str = "learning_objectives"
    estimated_minutes: int = 2

    async def render(self, content: dict[str, Any], format: BlockRenderFormat = BlockRenderFormat.MARKDOWN) -> str:
        """Render learning objectives as formatted list."""
        objectives = content.get("objectives", [])

        if format == BlockRenderFormat.MARKDOWN:
            lines = ["## Learning Objectives\n"]
            lines.extend(f"- {obj}" for obj in objectives)
            return "\n".join(lines)
        elif format == BlockRenderFormat.HTML:
            lines = ["<h2>Learning Objectives</h2>", "<ul>"]
            lines.extend(f"<li>{obj}</li>" for obj in objectives)
            lines.append("</ul>")
            return "\n".join(lines)
        elif format == BlockRenderFormat.LATEX:
            lines = ["\\section{Learning Objectives}", "\\begin{itemize}"]
            lines.extend(f"\\item {obj}" for obj in objectives)
            lines.append("\\end{itemize}")
            return "\n".join(lines)
        else:  # PLAIN_TEXT
            lines = ["LEARNING OBJECTIVES\n"]
            lines.extend(f"* {obj}" for obj in objectives)
            return "\n".join(lines)


class ConceptExplanationBlock(ContentBlock):
    """
    Theory and explanations of concepts.

    Scales with detail level - from brief definitions at low detail
    to comprehensive explanations with examples at high detail.
    """
    block_type: str = "concept_explanation"
    estimated_minutes: int = 5  # Base estimate, scales with content

    async def render(self, content: dict[str, Any], format: BlockRenderFormat = BlockRenderFormat.MARKDOWN) -> str:
        """Render concept explanations with appropriate formatting."""
        title = content.get("title", "Key Concepts")
        introduction = content.get("introduction", "")
        concepts = content.get("concepts", [])

        if format == BlockRenderFormat.MARKDOWN:
            lines = [f"## {title}\n"]
            if introduction:
                lines.append(f"{introduction}\n")

            for concept in concepts:
                lines.append(f"### {concept.get('name', 'Concept')}")
                lines.append(concept.get('explanation', ''))
                if concept.get('example'):
                    lines.append(f"\n**Example:** {concept['example']}")
                lines.append("")

            return "\n".join(lines)
        # TODO: Implement other formats
        return ""


class WorkedExampleBlock(ContentBlock):
    """
    Step-by-step solved problems.

    Each example includes the problem statement, detailed solution steps,
    and explanation of the reasoning. Typically takes ~5 minutes per example.
    """
    block_type: str = "worked_example"
    estimated_minutes: int = 5  # Per example

    async def render(self, content: dict[str, Any], format: BlockRenderFormat = BlockRenderFormat.MARKDOWN) -> str:
        """Render worked examples with solution steps."""
        examples = content.get("examples", [])

        if format == BlockRenderFormat.MARKDOWN:
            lines = ["## Worked Examples\n"]

            for i, example in enumerate(examples, 1):
                lines.append(f"### Example {i}")
                lines.append(f"**Problem:** {example.get('problem', '')}")
                lines.append("\n**Solution:**")

                steps = example.get('steps', [])
                for j, step in enumerate(steps, 1):
                    lines.append(f"{j}. {step}")

                if example.get('answer'):
                    lines.append(f"\n**Answer:** {example['answer']}")

                if example.get('explanation'):
                    lines.append(f"\n**Explanation:** {example['explanation']}")

                lines.append("")

            return "\n".join(lines)
        # TODO: Implement other formats
        return ""


class PracticeQuestionBlock(ContentBlock):
    """
    Questions for student practice.

    These are typically pulled from the question bank based on topic,
    difficulty, and other criteria. Can include hints and partial solutions.
    """
    block_type: str = "practice_questions"
    estimated_minutes: int = 3  # Per question average

    async def render(self, content: dict[str, Any], format: BlockRenderFormat = BlockRenderFormat.MARKDOWN) -> str:
        """Render practice questions from question bank or generated content."""
        questions = content.get("questions", [])
        include_answers = content.get("include_answers", False)

        if format == BlockRenderFormat.MARKDOWN:
            lines = ["## Practice Questions\n"]

            for i, q in enumerate(questions, 1):
                lines.append(f"**{i}.** {q.get('text', '')}")

                if q.get('marks'):
                    lines.append(f"*[{q['marks']} marks]*")

                if q.get('hint'):
                    lines.append(f"\n*Hint: {q['hint']}*")

                lines.append("")

            if include_answers and any(q.get('answer') for q in questions):
                lines.append("\n## Answer Key\n")
                for i, q in enumerate(questions, 1):
                    if q.get('answer'):
                        lines.append(f"**{i}.** {q['answer']}")
                        lines.append("")

            return "\n".join(lines)
        # TODO: Implement other formats
        return ""


class QuickReferenceBlock(ContentBlock):
    """
    Quick reference section with formulas, definitions, and key facts.

    Designed for quick lookup during problem-solving or review.
    Typically includes formulas, important definitions, and key relationships.
    """
    block_type: str = "quick_reference"
    estimated_minutes: int = 2

    async def render(self, content: dict[str, Any], format: BlockRenderFormat = BlockRenderFormat.MARKDOWN) -> str:
        """Render quick reference section."""
        formulas = content.get("formulas", [])
        definitions = content.get("definitions", [])
        facts = content.get("key_facts", [])

        if format == BlockRenderFormat.MARKDOWN:
            lines = ["## Quick Reference\n"]

            if formulas:
                lines.append("### Formulas")
                for formula in formulas:
                    lines.append(f"- {formula.get('name', '')}: `{formula.get('expression', '')}`")
                lines.append("")

            if definitions:
                lines.append("### Key Definitions")
                for defn in definitions:
                    lines.append(f"- **{defn.get('term', '')}**: {defn.get('definition', '')}")
                lines.append("")

            if facts:
                lines.append("### Important Facts")
                for fact in facts:
                    lines.append(f"- {fact}")
                lines.append("")

            return "\n".join(lines)
        # TODO: Implement other formats
        return ""


class SummaryBlock(ContentBlock):
    """
    Summary and key takeaways.

    Consolidates the main points of the material for review and retention.
    Can include summary points, key insights, and next steps.
    """
    block_type: str = "summary"
    estimated_minutes: int = 3

    async def render(self, content: dict[str, Any], format: BlockRenderFormat = BlockRenderFormat.MARKDOWN) -> str:
        """Render summary section."""
        key_points = content.get("key_points", [])
        insights = content.get("insights", [])
        next_steps = content.get("next_steps", [])

        if format == BlockRenderFormat.MARKDOWN:
            lines = ["## Summary\n"]

            if key_points:
                lines.append("### Key Points")
                for point in key_points:
                    lines.append(f"- {point}")
                lines.append("")

            if insights:
                lines.append("### Important Insights")
                for insight in insights:
                    lines.append(f"- {insight}")
                lines.append("")

            if next_steps:
                lines.append("### Next Steps")
                for step in next_steps:
                    lines.append(f"- {step}")
                lines.append("")

            return "\n".join(lines)
        # TODO: Implement other formats
        return ""


# Block registry for easy lookup
BLOCK_REGISTRY = {
    "learning_objectives": LearningObjectivesBlock,
    "concept_explanation": ConceptExplanationBlock,
    "worked_example": WorkedExampleBlock,
    "practice_questions": PracticeQuestionBlock,
    "quick_reference": QuickReferenceBlock,
    "summary": SummaryBlock,
}


def get_block_class(block_type: str) -> type[ContentBlock]:
    """Get block class by type identifier."""
    if block_type not in BLOCK_REGISTRY:
        raise ValueError(f"Unknown block type: {block_type}")
    return BLOCK_REGISTRY[block_type]


def estimate_document_time(blocks: list[dict[str, Any]]) -> int:
    """
    Estimate total time for a document based on its blocks.

    Args:
        blocks: List of block data with type and content

    Returns:
        Estimated time in minutes
    """
    total_minutes = 0

    for block_data in blocks:
        block_type = block_data.get("block_type")
        if not block_type:
            continue

        block_class = get_block_class(block_type)
        # Create instance to access estimated_minutes
        block_instance = block_class()
        base_minutes = block_instance.estimated_minutes

        # Adjust for content volume
        if block_type == "practice_questions":
            num_questions = len(block_data.get("content", {}).get("questions", []))
            total_minutes += base_minutes * num_questions
        elif block_type == "worked_example":
            num_examples = len(block_data.get("content", {}).get("examples", []))
            total_minutes += base_minutes * num_examples
        else:
            # For other blocks, use base estimate
            total_minutes += base_minutes

    return total_minutes
