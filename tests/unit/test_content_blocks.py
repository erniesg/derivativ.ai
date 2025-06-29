"""Unit tests for content block models."""

import pytest

from src.models.content_blocks import (
    BlockRenderFormat,
    ConceptExplanationBlock,
    LearningObjectivesBlock,
    PracticeQuestionBlock,
    QuickReferenceBlock,
    SummaryBlock,
    WorkedExampleBlock,
    estimate_document_time,
    get_block_class,
)


class TestLearningObjectivesBlock:
    """Test LearningObjectivesBlock rendering."""

    @pytest.mark.asyncio
    async def test_render_markdown(self):
        """Test markdown rendering of learning objectives."""
        block = LearningObjectivesBlock()
        content = {
            "objectives": [
                "Understand quadratic equations",
                "Apply the quadratic formula",
                "Solve real-world problems"
            ]
        }

        result = await block.render(content, BlockRenderFormat.MARKDOWN)

        assert "## Learning Objectives" in result
        assert "- Understand quadratic equations" in result
        assert "- Apply the quadratic formula" in result
        assert "- Solve real-world problems" in result

    @pytest.mark.asyncio
    async def test_render_html(self):
        """Test HTML rendering of learning objectives."""
        block = LearningObjectivesBlock()
        content = {
            "objectives": [
                "Understand vectors",
                "Calculate dot products"
            ]
        }

        result = await block.render(content, BlockRenderFormat.HTML)

        assert "<h2>Learning Objectives</h2>" in result
        assert "<ul>" in result
        assert "<li>Understand vectors</li>" in result
        assert "<li>Calculate dot products</li>" in result
        assert "</ul>" in result

    @pytest.mark.asyncio
    async def test_render_latex(self):
        """Test LaTeX rendering of learning objectives."""
        block = LearningObjectivesBlock()
        content = {"objectives": ["Master derivatives"]}

        result = await block.render(content, BlockRenderFormat.LATEX)

        assert "\\section{Learning Objectives}" in result
        assert "\\begin{itemize}" in result
        assert "\\item Master derivatives" in result
        assert "\\end{itemize}" in result

    @pytest.mark.asyncio
    async def test_empty_objectives(self):
        """Test rendering with no objectives."""
        block = LearningObjectivesBlock()
        content = {"objectives": []}

        result = await block.render(content, BlockRenderFormat.MARKDOWN)

        assert "## Learning Objectives" in result
        assert result.strip().endswith("Learning Objectives")


class TestConceptExplanationBlock:
    """Test ConceptExplanationBlock rendering."""

    @pytest.mark.asyncio
    async def test_render_markdown_full_content(self):
        """Test markdown rendering with all content fields."""
        block = ConceptExplanationBlock()
        content = {
            "title": "Quadratic Equations",
            "introduction": "Quadratic equations are polynomial equations of degree 2.",
            "concepts": [
                {
                    "name": "Standard Form",
                    "explanation": "A quadratic equation in standard form is ax² + bx + c = 0",
                    "example": "2x² + 5x - 3 = 0"
                },
                {
                    "name": "Factoring Method",
                    "explanation": "Some quadratics can be factored into linear terms",
                    "example": "x² - 5x + 6 = (x - 2)(x - 3)"
                }
            ]
        }

        result = await block.render(content, BlockRenderFormat.MARKDOWN)

        assert "## Quadratic Equations" in result
        assert "polynomial equations of degree 2" in result
        assert "### Standard Form" in result
        assert "ax² + bx + c = 0" in result
        assert "**Example:** 2x² + 5x - 3 = 0" in result
        assert "### Factoring Method" in result

    @pytest.mark.asyncio
    async def test_render_markdown_minimal(self):
        """Test markdown rendering with minimal content."""
        block = ConceptExplanationBlock()
        content = {
            "concepts": [
                {"name": "Simple Concept", "explanation": "Basic explanation"}
            ]
        }

        result = await block.render(content, BlockRenderFormat.MARKDOWN)

        assert "## Key Concepts" in result  # Default title
        assert "### Simple Concept" in result
        assert "Basic explanation" in result


class TestWorkedExampleBlock:
    """Test WorkedExampleBlock rendering."""

    @pytest.mark.asyncio
    async def test_render_markdown_complete_example(self):
        """Test rendering a complete worked example."""
        block = WorkedExampleBlock()
        content = {
            "examples": [
                {
                    "problem": "Solve x² + 5x + 6 = 0",
                    "steps": [
                        "Identify a=1, b=5, c=6",
                        "Calculate discriminant: b² - 4ac = 25 - 24 = 1",
                        "Apply quadratic formula: x = (-5 ± √1) / 2",
                        "Simplify: x = (-5 ± 1) / 2"
                    ],
                    "answer": "x = -2 or x = -3",
                    "explanation": "We used the quadratic formula since the discriminant is positive."
                }
            ]
        }

        result = await block.render(content, BlockRenderFormat.MARKDOWN)

        assert "## Worked Examples" in result
        assert "### Example 1" in result
        assert "**Problem:** Solve x² + 5x + 6 = 0" in result
        assert "**Solution:**" in result
        assert "1. Identify a=1, b=5, c=6" in result
        assert "4. Simplify: x = (-5 ± 1) / 2" in result
        assert "**Answer:** x = -2 or x = -3" in result
        assert "**Explanation:** We used the quadratic formula" in result

    @pytest.mark.asyncio
    async def test_render_multiple_examples(self):
        """Test rendering multiple examples."""
        block = WorkedExampleBlock()
        content = {
            "examples": [
                {"problem": "Problem 1", "steps": ["Step 1"], "answer": "Answer 1"},
                {"problem": "Problem 2", "steps": ["Step A", "Step B"], "answer": "Answer 2"}
            ]
        }

        result = await block.render(content, BlockRenderFormat.MARKDOWN)

        assert "### Example 1" in result
        assert "### Example 2" in result
        assert "Problem 1" in result
        assert "Problem 2" in result


class TestPracticeQuestionBlock:
    """Test PracticeQuestionBlock rendering."""

    @pytest.mark.asyncio
    async def test_render_questions_without_answers(self):
        """Test rendering practice questions without answers."""
        block = PracticeQuestionBlock()
        content = {
            "questions": [
                {"text": "Solve 2x + 3 = 7", "marks": 2},
                {"text": "Factorize x² - 9", "marks": 3, "hint": "Use difference of squares"}
            ],
            "include_answers": False
        }

        result = await block.render(content, BlockRenderFormat.MARKDOWN)

        assert "## Practice Questions" in result
        assert "**1.** Solve 2x + 3 = 7" in result
        assert "*[2 marks]*" in result
        assert "**2.** Factorize x² - 9" in result
        assert "*[3 marks]*" in result
        assert "*Hint: Use difference of squares*" in result
        assert "Answer Key" not in result

    @pytest.mark.asyncio
    async def test_render_questions_with_answers(self):
        """Test rendering practice questions with answer key."""
        block = PracticeQuestionBlock()
        content = {
            "questions": [
                {"text": "Solve 2x + 3 = 7", "marks": 2, "answer": "x = 2"},
                {"text": "Factorize x² - 9", "marks": 3, "answer": "(x + 3)(x - 3)"}
            ],
            "include_answers": True
        }

        result = await block.render(content, BlockRenderFormat.MARKDOWN)

        assert "## Practice Questions" in result
        assert "## Answer Key" in result
        assert "**1.** x = 2" in result
        assert "**2.** (x + 3)(x - 3)" in result


class TestQuickReferenceBlock:
    """Test QuickReferenceBlock rendering."""

    @pytest.mark.asyncio
    async def test_render_all_sections(self):
        """Test rendering with formulas, definitions, and facts."""
        block = QuickReferenceBlock()
        content = {
            "formulas": [
                {"name": "Quadratic Formula", "expression": "x = (-b ± √(b² - 4ac)) / 2a"},
                {"name": "Discriminant", "expression": "Δ = b² - 4ac"}
            ],
            "definitions": [
                {"term": "Quadratic", "definition": "A polynomial of degree 2"},
                {"term": "Root", "definition": "A value of x that makes the equation equal to zero"}
            ],
            "key_facts": [
                "If Δ > 0, the equation has two distinct real roots",
                "If Δ = 0, the equation has one repeated real root"
            ]
        }

        result = await block.render(content, BlockRenderFormat.MARKDOWN)

        assert "## Quick Reference" in result
        assert "### Formulas" in result
        assert "- Quadratic Formula: `x = (-b ± √(b² - 4ac)) / 2a`" in result
        assert "### Key Definitions" in result
        assert "- **Quadratic**: A polynomial of degree 2" in result
        assert "### Important Facts" in result
        assert "- If Δ > 0, the equation has two distinct real roots" in result

    @pytest.mark.asyncio
    async def test_render_partial_content(self):
        """Test rendering with only some sections present."""
        block = QuickReferenceBlock()
        content = {
            "formulas": [{"name": "Area of Circle", "expression": "A = πr²"}],
            "definitions": [],  # Empty
            "key_facts": None  # Missing
        }

        result = await block.render(content, BlockRenderFormat.MARKDOWN)

        assert "### Formulas" in result
        assert "Area of Circle" in result
        assert "### Key Definitions" not in result
        assert "### Important Facts" not in result


class TestSummaryBlock:
    """Test SummaryBlock rendering."""

    @pytest.mark.asyncio
    async def test_render_complete_summary(self):
        """Test rendering with all summary sections."""
        block = SummaryBlock()
        content = {
            "key_points": [
                "Quadratic equations have degree 2",
                "The discriminant determines the nature of roots"
            ],
            "insights": [
                "Factoring is fastest when applicable",
                "The quadratic formula always works"
            ],
            "next_steps": [
                "Practice solving various types of quadratics",
                "Learn about complex roots"
            ]
        }

        result = await block.render(content, BlockRenderFormat.MARKDOWN)

        assert "## Summary" in result
        assert "### Key Points" in result
        assert "- Quadratic equations have degree 2" in result
        assert "### Important Insights" in result
        assert "- Factoring is fastest when applicable" in result
        assert "### Next Steps" in result
        assert "- Learn about complex roots" in result


class TestBlockRegistry:
    """Test block registry functions."""

    def test_get_block_class_valid(self):
        """Test getting valid block classes."""
        assert get_block_class("learning_objectives") == LearningObjectivesBlock
        assert get_block_class("concept_explanation") == ConceptExplanationBlock
        assert get_block_class("worked_example") == WorkedExampleBlock
        assert get_block_class("practice_questions") == PracticeQuestionBlock
        assert get_block_class("quick_reference") == QuickReferenceBlock
        assert get_block_class("summary") == SummaryBlock

    def test_get_block_class_invalid(self):
        """Test getting invalid block class raises error."""
        with pytest.raises(ValueError, match="Unknown block type: invalid_block"):
            get_block_class("invalid_block")


class TestTimeEstimation:
    """Test document time estimation."""

    def test_estimate_document_time_basic(self):
        """Test basic time estimation for blocks."""
        blocks = [
            {"block_type": "learning_objectives", "content": {}},  # 2 min
            {"block_type": "concept_explanation", "content": {}},  # 5 min
            {"block_type": "summary", "content": {}}  # 3 min
        ]

        assert estimate_document_time(blocks) == 10

    def test_estimate_document_time_with_questions(self):
        """Test time estimation with practice questions."""
        blocks = [
            {"block_type": "learning_objectives", "content": {}},  # 2 min
            {
                "block_type": "practice_questions",
                "content": {
                    "questions": [
                        {"text": "Q1"},
                        {"text": "Q2"},
                        {"text": "Q3"}
                    ]
                }
            }  # 3 questions × 3 min = 9 min
        ]

        assert estimate_document_time(blocks) == 11

    def test_estimate_document_time_with_examples(self):
        """Test time estimation with worked examples."""
        blocks = [
            {
                "block_type": "worked_example",
                "content": {
                    "examples": [
                        {"problem": "P1"},
                        {"problem": "P2"}
                    ]
                }
            }  # 2 examples × 5 min = 10 min
        ]

        assert estimate_document_time(blocks) == 10

    def test_estimate_document_time_missing_type(self):
        """Test time estimation handles missing block type."""
        blocks = [
            {"block_type": "summary", "content": {}},  # 3 min
            {"content": {}},  # No block_type, ignored
            {"block_type": None, "content": {}}  # None type, ignored
        ]

        assert estimate_document_time(blocks) == 3
