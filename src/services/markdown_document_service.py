"""
Markdown-First Document Generation Service

Generates clean, structured markdown documents that can be converted to
multiple formats using pandoc, eliminating complex JSON structure issues.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from src.core.config import get_settings
from src.models.llm_models import LLMRequest
from src.models.markdown_generation_models import MarkdownGenerationRequest
from src.services.llm_service import LLMService
from src.services.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class MarkdownDocumentService:
    """Service for generating clean markdown documents."""

    def __init__(self, llm_service: LLMService, prompt_manager: PromptManager):
        self.llm_service = llm_service
        self.prompt_manager = prompt_manager
        self.settings = get_settings()

    async def generate_markdown_document(
        self,
        request: MarkdownGenerationRequest,
        custom_instructions: Optional[str] = None
    ) -> dict[str, Any]:
        """Generate a clean markdown document.

        Returns:
            {
                "success": bool,
                "markdown_content": str,  # Clean markdown text
                "metadata": dict,        # Document metadata
                "generation_info": dict  # Generation details
            }
        """
        try:
            logger.info(f"🎯 Generating markdown document: {request.document_type} - {request.topic}")

            # Generate clean markdown content
            markdown_content = await self._generate_markdown_content(request, custom_instructions)

            # Extract metadata
            metadata = self._extract_document_metadata(request, markdown_content)

            # Generation info
            generation_info = {
                "generated_at": datetime.now().isoformat(),
                "document_type": request.document_type.value,
                "topic": request.topic.value,
                "tier": request.tier.value,
                "detail_level": request.detail_level,
                "target_duration": request.target_duration_minutes,
                "grade_level": request.grade_level
            }

            logger.info(f"✅ Generated {len(markdown_content)} characters of markdown content")

            return {
                "success": True,
                "markdown_content": markdown_content,
                "metadata": metadata,
                "generation_info": generation_info
            }

        except Exception as e:
            logger.error(f"❌ Markdown generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "markdown_content": "",
                "metadata": {},
                "generation_info": {}
            }

    async def _generate_markdown_content(
        self,
        request: MarkdownGenerationRequest,
        custom_instructions: Optional[str]
    ) -> str:
        """Generate clean markdown content using LLM."""

        # Use markdown-optimized template with request context
        template_context = request.to_template_context()
        if custom_instructions:
            template_context["custom_instructions"] = custom_instructions

        # Simple template rendering (bypass PromptManager for now)
        prompt = self._render_simple_template(template_context)

        # Generate with primary LLM
        llm_request = LLMRequest(
            model="gpt-4o",  # Use GPT-4o for high-quality markdown generation
            prompt=prompt,
            temperature=0.7,
            max_tokens=4000,
            stream=False  # Don't stream for this use case
        )

        response = await self.llm_service.generate(llm_request)

        # If we get here, generation was successful (no exception thrown)
        # Clean and validate markdown
        markdown_content = self._clean_markdown_content(response.content)

        return markdown_content

    def _clean_markdown_content(self, raw_content: str) -> str:
        """Clean and validate markdown content."""

        # Remove any JSON artifacts if LLM returned JSON instead of markdown
        content = raw_content.strip()

        # If content starts with ```markdown, extract the markdown block
        if content.startswith("```markdown"):
            lines = content.split("\n")
            start_idx = 1  # Skip ```markdown line
            end_idx = len(lines)

            # Find closing ```
            for i, line in enumerate(lines[1:], 1):
                if line.strip() == "```":
                    end_idx = i
                    break

            content = "\n".join(lines[start_idx:end_idx])

        # Basic markdown validation and cleanup
        lines = content.split("\n")
        cleaned_lines = []

        for line in lines:
            # Skip empty lines at start
            if not cleaned_lines and not line.strip():
                continue

            # Clean up common issues
            line = line.replace("**Learning Objectives:**", "## Learning Objectives")
            line = line.replace("**Practice Questions:**", "## Practice Questions")
            line = line.replace("**Worked Examples:**", "## Worked Examples")
            line = line.replace("**Key Formulas:**", "## Key Formulas")
            line = line.replace("**Solutions:**", "## Solutions")

            cleaned_lines.append(line)

        # Ensure proper title structure
        if cleaned_lines and not cleaned_lines[0].startswith("#"):
            # Add main title if missing
            cleaned_lines.insert(0, "# Mathematics Worksheet")
            cleaned_lines.insert(1, "")

        return "\n".join(cleaned_lines)

    def _render_simple_template(self, context: dict) -> str:
        """Simple template rendering for markdown generation."""

        # Extract context variables
        document_type = context.get("document_type", "worksheet")
        topic = context.get("topic", "mathematics")
        tier = context.get("tier", "Core")
        detail_level = context.get("detail_level", 5)
        target_duration = context.get("target_duration_minutes", 30)
        grade_level = context.get("grade_level", "7-9")
        custom_instructions = context.get("custom_instructions", "")

        # Build prompt components
        base_prompt = "You are an expert educational content creator specializing in Cambridge IGCSE Mathematics."
        task_description = f"Generate a clean, well-structured markdown document for a {document_type} about {topic}."

        requirements = f"""## Document Requirements
- Document Type: {document_type.title()}
- Topic: {topic.title()}
- Tier: {tier}
- Detail Level: {detail_level}/10
- Target Duration: {target_duration} minutes
- Grade Level: {grade_level}"""

        if custom_instructions:
            requirements += f"\n\n## Custom Instructions\n{custom_instructions}"

        structure = """## Output Requirements

Generate a clean, professional markdown document with the following structure:

1. Main Title (H1) - Clear, engaging title
2. Learning Objectives (H2) - 3-5 clear learning goals
3. Key Concepts (H2) - Essential definitions and explanations
4. Worked Examples (H2) - Step-by-step solutions with reasoning
5. Practice Questions (H2) - Graduated difficulty questions
6. Solutions (H2) - Detailed solutions for practice questions"""

        guidelines = f"""## Content Guidelines

### Mathematical Content Standards:
- Use clear mathematical notation
- Show all working steps in solutions
- Ensure mathematical accuracy and grade-appropriate difficulty

### Formatting Guidelines:
- Use proper markdown headers (# ## ###)
- Use bullet points (-) for lists
- Use numbered lists (1. 2. 3.) for sequential steps
- Use bold text for key terms and important points

## Topic-Specific Requirements for {topic.title()}:

Generate content that covers essential {topic} concepts appropriate for {tier} tier students, including:
- Core mathematical principles and definitions
- Practical problem-solving techniques
- Real-world applications where relevant

## Time Allocation ({target_duration} minutes total):
- Learning objectives review: 3 minutes
- Concept explanation: {int(target_duration * 0.3)} minutes
- Worked examples: {int(target_duration * 0.4)} minutes
- Practice questions: {int(target_duration * 0.3)} minutes"""

        final_instruction = "Generate the complete markdown document now, following the structure and guidelines above. Return ONLY the markdown content, no additional formatting or explanations."

        # Combine all parts
        prompt = f"{base_prompt} {task_description}\n\n{requirements}\n\n{structure}\n\n{guidelines}\n\n{final_instruction}"

        return prompt

    def _extract_document_metadata(
        self,
        request: MarkdownGenerationRequest,
        markdown_content: str
    ) -> dict[str, Any]:
        """Extract metadata from request and content."""

        # Count content sections
        lines = markdown_content.split("\n")
        sections = [line for line in lines if line.startswith("##")]

        # Estimate reading time (200 words per minute)
        word_count = len(markdown_content.split())
        estimated_reading_minutes = max(1, word_count // 200)

        return {
            "title": request.get_title(),
            "topic": request.topic.value,
            "tier": request.tier.value,
            "detail_level": request.detail_level.value,
            "grade_level": request.grade_level,
            "target_duration_minutes": request.target_duration_minutes,
            "estimated_reading_minutes": estimated_reading_minutes,
            "word_count": word_count,
            "section_count": len(sections),
            "sections": [section.replace("##", "").strip() for section in sections]
        }

    def preview_markdown_as_html(self, markdown_content: str) -> str:
        """Convert markdown to HTML for preview (simple conversion)."""
        try:
            import markdown

            html = markdown.markdown(
                markdown_content,
                extensions=['extra', 'codehilite']
            )

            # Add basic styling
            styled_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Document Preview</title>
                <style>
                    body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
                    h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                    h2 {{ color: #27ae60; margin-top: 30px; }}
                    h3 {{ color: #e67e22; }}
                    code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
                    blockquote {{ border-left: 4px solid #3498db; padding-left: 20px; margin-left: 0; }}
                    .math {{ background: #f9f9f9; padding: 10px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                {html}
            </body>
            </html>
            """

            return styled_html

        except ImportError:
            # Fallback: basic HTML structure
            html_content = markdown_content.replace("\n", "<br>")
            html_content = html_content.replace("# ", "<h1>")
            html_content = html_content.replace("## ", "<h2>")
            html_content = html_content.replace("### ", "<h3>")

            return f"<html><body>{html_content}</body></html>"
