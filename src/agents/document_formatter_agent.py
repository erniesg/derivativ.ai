"""
Document Formatter Agent for transforming content into various output formats.

Specializes in converting generated document content into properly formatted
HTML, PDF, DOCX, LaTeX, Markdown, and presentation formats.
"""

import logging
from typing import Any

from src.agents.base_agent import BaseAgent
from src.models.document_models import (
    ContentSection,
    DocumentType,
    ExportFormat,
    GeneratedDocument,
)
from src.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class DocumentFormatterAgent(BaseAgent):
    """
    Agent responsible for formatting and transforming document content.

    Converts structured document data into various output formats
    with appropriate styling, layout, and formatting rules.
    """

    def __init__(self, llm_service: LLMService, name: str = "DocumentFormatter"):
        super().__init__(llm_service, name)
        self.formatting_templates = self._load_formatting_templates()

    def _load_formatting_templates(self) -> dict[str, dict[str, str]]:
        """Load formatting templates for different output formats."""
        return {
            "html": {
                "document_header": """
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>{title}</title>
                    <style>
                        body {{
                            font-family: 'Times New Roman', serif;
                            line-height: 1.6;
                            max-width: 800px;
                            margin: 0 auto;
                            padding: 20px;
                            color: #333;
                        }}
                        .document-header {{
                            text-align: center;
                            border-bottom: 2px solid #333;
                            padding-bottom: 20px;
                            margin-bottom: 30px;
                        }}
                        .section {{
                            margin: 30px 0;
                            page-break-inside: avoid;
                        }}
                        .question {{
                            margin: 20px 0;
                            padding: 15px;
                            border-left: 4px solid #007acc;
                            background-color: #f8f9fa;
                        }}
                        .solution {{
                            background-color: #e8f4f8;
                            padding: 15px;
                            margin: 15px 0;
                            border-radius: 5px;
                        }}
                        .marks {{
                            float: right;
                            font-weight: bold;
                            color: #d63384;
                        }}
                        .worked-example {{
                            border: 1px solid #dee2e6;
                            padding: 20px;
                            margin: 20px 0;
                            background-color: #fff;
                        }}
                        h1 {{ color: #2c3e50; }}
                        h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
                        h3 {{ color: #7f8c8d; }}
                        .page-break {{ page-break-before: always; }}
                        @media print {{
                            body {{ font-size: 12pt; }}
                            .question, .solution, .worked-example {{ break-inside: avoid; }}
                        }}
                    </style>
                </head>
                <body>
                """,
                "document_footer": """
                </body>
                </html>
                """,
                "section_template": """
                <div class="section">
                    <h2>{title}</h2>
                    {content}
                </div>
                """,
                "question_template": """
                <div class="question">
                    <span class="marks">[{marks} marks]</span>
                    <div><strong>Question {number}:</strong></div>
                    <div>{question_text}</div>
                </div>
                """,
                "solution_template": """
                <div class="solution">
                    <strong>Solution {number}:</strong>
                    {solution_content}
                </div>
                """,
            },
            "latex": {
                "document_header": """
                \\documentclass[12pt,a4paper]{{article}}
                \\usepackage[utf8]{{inputenc}}
                \\usepackage[T1]{{fontenc}}
                \\usepackage{{amsmath,amssymb,amsthm}}
                \\usepackage{{geometry}}
                \\usepackage{{fancyhdr}}
                \\usepackage{{enumitem}}
                \\usepackage{{graphicx}}
                \\geometry{{margin=2.5cm}}
                \\pagestyle{{fancy}}
                \\fancyhf{{}}
                \\fancyhead[C]{{{title}}}
                \\fancyfoot[C]{{\\thepage}}
                \\title{{{title}}}
                \\author{{Generated by Derivativ AI}}
                \\date{{\\today}}
                \\begin{{document}}
                \\maketitle
                """,
                "document_footer": """
                \\end{document}
                """,
                "section_template": """
                \\section{{{title}}}
                {content}
                """,
                "question_template": """
                \\begin{{enumerate}}[resume]
                \\item {question_text} \\hfill [{marks} marks]
                \\end{{enumerate}}
                """,
                "solution_template": """
                \\textbf{{Solution:}} {solution_content}
                """,
            },
            "markdown": {
                "document_header": "# {title}\n\n*Generated by Derivativ AI*\n\n",
                "section_template": "## {title}\n\n{content}\n\n",
                "question_template": "**Question {number}** ({marks} marks)\n\n{question_text}\n\n",
                "solution_template": "**Solution {number}:**\n\n{solution_content}\n\n",
            },
        }

    async def _execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute document formatting process."""
        self._observe("Starting document formatting", input_data)

        try:
            # Parse input data
            format_request = self._parse_format_request(input_data)

            self._think(f"Formatting document to {format_request['format']} format")

            # Generate formatted content
            formatted_content = await self._format_document(format_request)

            self._act("Document formatting completed", {"format": format_request["format"]})

            return {
                "formatted_content": formatted_content,
                "format": format_request["format"],
                "success": True,
            }

        except Exception as e:
            logger.error(f"Document formatting failed: {e}")
            return {"success": False, "error": str(e)}

    def _parse_format_request(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Parse and validate format request."""
        required_fields = ["document", "format"]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")

        document = input_data["document"]
        if not isinstance(document, GeneratedDocument):
            raise ValueError("Document must be a GeneratedDocument instance")

        format_type = input_data["format"]
        if format_type not in ExportFormat.__members__.values():
            raise ValueError(f"Unsupported format: {format_type}")

        return {
            "document": document,
            "format": format_type,
            "options": input_data.get("options", {}),
        }

    async def _format_document(self, format_request: dict[str, Any]) -> str:
        """Format document according to specified format."""
        document = format_request["document"]
        format_type = format_request["format"]
        options = format_request.get("options", {})

        # Apply personalization from document or export options
        personalization = self._extract_personalization(document, options)

        if format_type == ExportFormat.HTML:
            return self._format_to_html(document, personalization)
        elif format_type == ExportFormat.LATEX:
            return self._format_to_latex(document, personalization)
        elif format_type == ExportFormat.MARKDOWN:
            return self._format_to_markdown(document, personalization)
        elif format_type == ExportFormat.PDF:
            # PDF generation would require additional tools (weasyprint, reportlab, etc.)
            return await self._format_to_pdf(document, personalization)
        elif format_type == ExportFormat.DOCX:
            # DOCX generation would require python-docx
            return await self._format_to_docx(document, personalization)
        elif format_type == ExportFormat.SLIDES_PPTX:
            # PowerPoint generation for slides
            return await self._format_to_slides(document, personalization)
        else:
            raise ValueError(f"Format {format_type} not yet implemented")

    def _format_to_html(
        self, document: GeneratedDocument, personalization: dict[str, Any] = None
    ) -> str:
        """Format document as HTML with personalization."""
        templates = self.formatting_templates["html"]
        personalization = personalization or {}

        # Customize CSS based on personalization
        custom_css = self._generate_custom_css(personalization)

        # Start with header
        html = templates["document_header"].format(title=document.title)

        # Insert custom CSS before closing head tag
        if custom_css:
            html = html.replace("</style>", f"{custom_css}</style>")

        # Add document metadata
        html += f"""
        <div class="document-header">
            <h1>{document.title}</h1>
            <p><em>{document.document_type.value.title()} - {document.detail_level.value.title()} Level</em></p>
            <p>Total Questions: {document.total_questions} | Estimated Time: {document.estimated_duration} minutes</p>
        """

        # Add personalization notice if customizations were applied
        if document.applied_customizations:
            html += "<p class='personalization-notice'><small>✨ This document has been personalized based on your preferences</small></p>"

        html += "</div>"

        # Add sections with personalized formatting
        for section in document.sections:
            section_content = self._format_section_html(section, templates, personalization)
            html += templates["section_template"].format(
                title=section.title, content=section_content
            )

        # Add footer
        html += templates["document_footer"]

        return html

    def _format_section_html(
        self,
        section: ContentSection,
        templates: dict[str, str],
        personalization: dict[str, Any] = None,
    ) -> str:
        """Format a specific section as HTML with personalization."""
        content = ""
        personalization = personalization or {}

        if section.content_type == "practice_questions":
            content += self._format_questions_html(section, templates, personalization)
        elif section.content_type == "solutions":
            content += self._format_solutions_html(section, templates, personalization)
        elif section.content_type == "worked_examples":
            content += self._format_worked_examples_html(section, templates, personalization)
        elif section.content_type == "learning_objectives":
            objectives_text = section.content_data.get("objectives_text", "")
            content += f"<div class='learning-objectives'>{objectives_text}</div>"
        else:
            # Generic text content
            text_content = section.content_data.get("text", "")
            content += f"<p>{text_content}</p>"

        # Add subsections
        for subsection in section.subsections:
            subsection_content = self._format_section_html(subsection, templates, personalization)
            content += (
                f"<div class='subsection'><h3>{subsection.title}</h3>{subsection_content}</div>"
            )

        return content

    def _format_questions_html(
        self,
        section: ContentSection,
        templates: dict[str, str],
        personalization: dict[str, Any] = None,
    ) -> str:
        """Format practice questions as HTML with personalization."""
        content = ""
        questions = section.content_data.get("questions", [])
        personalization = personalization or {}

        # Add visual cues for visual learners
        visual_class = (
            "visual-enhanced" if personalization.get("learning_style") == "visual" else ""
        )

        for i, question in enumerate(questions, 1):
            question_html = templates["question_template"].format(
                number=i,
                question_text=question.get("question_text", ""),
                marks=question.get("marks", 0),
            )

            # Add visual enhancement wrapper if needed
            if visual_class:
                question_html = f"<div class='{visual_class}'>{question_html}</div>"

            content += question_html

        if section.content_data.get("total_marks"):
            content += f"<div class='total-marks'><strong>Total: {section.content_data['total_marks']} marks</strong></div>"

        return content

    def _format_solutions_html(
        self,
        section: ContentSection,
        templates: dict[str, str],
        personalization: dict[str, Any] = None,
    ) -> str:
        """Format solutions as HTML."""
        content = ""
        solutions = section.content_data.get("solutions", [])

        for solution in solutions:
            solution_content = ""

            # Add solution steps
            for step in solution.get("solution_steps", []):
                solution_content += f"<p><strong>Step {step.get('step_number', '')}:</strong> {step.get('description_text', '')}</p>"

            # Add final answer
            for answer in solution.get("final_answers", []):
                solution_content += (
                    f"<p><strong>Answer:</strong> {answer.get('answer_text', '')}</p>"
                )

            content += templates["solution_template"].format(
                number=solution.get("question_number", ""), solution_content=solution_content
            )

        return content

    def _format_worked_examples_html(
        self,
        section: ContentSection,
        templates: dict[str, str],
        personalization: dict[str, Any] = None,
    ) -> str:
        """Format worked examples as HTML."""
        content = ""
        examples = section.content_data.get("examples", [])

        for i, example in enumerate(examples, 1):
            example_content = f"""
            <div class="worked-example">
                <h4>Example {i}</h4>
                <div class="example-question">
                    <strong>Question:</strong> {example.get('question_text', '')}
                    <span class="marks">[{example.get('marks', 0)} marks]</span>
                </div>
                <div class="example-solution">
                    <strong>Solution:</strong>
            """

            for step in example.get("solution_steps", []):
                example_content += f"<p>{step.get('description_text', '')}</p>"

            example_content += "</div></div>"
            content += example_content

        return content

    def _format_to_latex(
        self, document: GeneratedDocument, personalization: dict[str, Any] = None
    ) -> str:
        """Format document as LaTeX."""
        templates = self.formatting_templates["latex"]

        latex = templates["document_header"].format(title=document.title)

        for section in document.sections:
            section_content = self._format_section_latex(section, templates)
            latex += templates["section_template"].format(
                title=section.title, content=section_content
            )

        latex += templates["document_footer"]
        return latex

    def _format_section_latex(self, section: ContentSection, templates: dict[str, str]) -> str:
        """Format section as LaTeX."""
        content = ""

        if section.content_type == "practice_questions":
            questions = section.content_data.get("questions", [])
            for question in questions:
                content += templates["question_template"].format(
                    question_text=question.get("question_text", "").replace("$", "\\$"),
                    marks=question.get("marks", 0),
                )
        else:
            text_content = section.content_data.get("text", "")
            content += text_content.replace("$", "\\$").replace("&", "\\&")

        return content

    def _format_to_markdown(
        self, document: GeneratedDocument, personalization: dict[str, Any] = None
    ) -> str:
        """Format document as Markdown with personalization."""
        templates = self.formatting_templates["markdown"]
        personalization = personalization or {}

        markdown = templates["document_header"].format(title=document.title)
        markdown += f"**Document Type:** {document.document_type.value}  \n"
        markdown += f"**Detail Level:** {document.detail_level.value}  \n"
        markdown += f"**Total Questions:** {document.total_questions}  \n"

        # Add personalization info if available
        if document.applied_customizations:
            markdown += "**Personalized:** Yes ✨  \n"

        markdown += "\n"

        for section in document.sections:
            section_content = self._format_section_markdown(section, templates, personalization)
            markdown += templates["section_template"].format(
                title=section.title, content=section_content
            )

        return markdown

    def _format_section_markdown(
        self,
        section: ContentSection,
        templates: dict[str, str],
        personalization: dict[str, Any] = None,
    ) -> str:
        """Format section as Markdown with personalization."""
        content = ""
        personalization = personalization or {}

        if section.content_type == "practice_questions":
            questions = section.content_data.get("questions", [])
            for i, question in enumerate(questions, 1):
                question_content = templates["question_template"].format(
                    number=i,
                    question_text=question.get("question_text", ""),
                    marks=question.get("marks", 0),
                )

                # Add visual hints for visual learners
                if personalization.get("learning_style") == "visual":
                    question_content += (
                        "\n> 💡 *Consider drawing a diagram to visualize this problem*\n"
                    )

                content += question_content
        elif section.content_type == "learning_objectives":
            content += section.content_data.get("objectives_text", "")
        else:
            content += section.content_data.get("text", "")

        return content

    async def _format_to_pdf(
        self, document: GeneratedDocument, personalization: dict[str, Any] = None
    ) -> str:
        """Format document as PDF (placeholder for future implementation)."""
        # This would require integration with libraries like:
        # - weasyprint (HTML to PDF)
        # - reportlab (programmatic PDF generation)
        # - LaTeX compiler (pdflatex)

        self._think("PDF generation requires additional dependencies")

        # For now, return HTML that can be converted to PDF
        html_content = self._format_to_html(document)

        # In a full implementation, this would:
        # 1. Use weasyprint to convert HTML to PDF
        # 2. Return the PDF file path or binary data

        return f"PDF generation placeholder - HTML content ready for conversion:\n{html_content}"

    async def _format_to_docx(
        self, document: GeneratedDocument, personalization: dict[str, Any] = None
    ) -> str:
        """Format document as DOCX (placeholder for future implementation)."""
        # This would require python-docx library

        self._think("DOCX generation requires python-docx integration")

        # In a full implementation, this would:
        # 1. Use python-docx to create Word document
        # 2. Add sections, questions, formatting
        # 3. Return the file path or binary data

        return f"DOCX generation placeholder for document: {document.title}"

    async def _format_to_slides(
        self, document: GeneratedDocument, personalization: dict[str, Any] = None
    ) -> str:
        """Format document as PowerPoint slides (placeholder for future implementation)."""
        # This would require python-pptx library

        self._think("PowerPoint generation requires python-pptx integration")

        if document.document_type != DocumentType.SLIDES:
            logger.warning("Converting non-slide document to presentation format")

        # In a full implementation, this would:
        # 1. Use python-pptx to create presentation
        # 2. Create slides for each section
        # 3. Add appropriate layouts and formatting
        # 4. Return the file path or binary data

        return f"PowerPoint generation placeholder for: {document.title}"

    def _extract_personalization(
        self, document: GeneratedDocument, export_options: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract personalization settings from document and export options."""
        personalization = {}

        # Get personalization from document's applied customizations
        if (
            document.applied_customizations
            and "personalization_context" in document.applied_customizations
        ):
            personalization.update(document.applied_customizations["personalization_context"])

        # Override with export-specific personalization
        if "export_personalization" in export_options:
            personalization.update(export_options["export_personalization"])

        return personalization

    def _generate_custom_css(self, personalization: dict[str, Any]) -> str:
        """Generate custom CSS based on personalization preferences."""
        custom_css = ""

        if personalization.get("learning_style") == "visual":
            custom_css += """
            .visual-enhanced {
                border-left: 5px solid #007acc;
                background: linear-gradient(90deg, #f8f9fa 0%, #e3f2fd 100%);
                padding: 15px;
                margin: 10px 0;
            }
            .personalization-notice {
                background-color: #e8f5e8;
                padding: 8px;
                border-radius: 4px;
                border-left: 4px solid #4caf50;
            }
            """

        if personalization.get("font_size") == "large":
            custom_css += """
            body { font-size: 18px; }
            h1 { font-size: 2.5em; }
            h2 { font-size: 2em; }
            h3 { font-size: 1.5em; }
            """

        if personalization.get("high_contrast"):
            custom_css += """
            body { background-color: #000; color: #fff; }
            .question { background-color: #333; color: #fff; }
            .solution { background-color: #222; color: #fff; }
            """

        return custom_css
