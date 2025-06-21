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
        elif format_type == ExportFormat.MARKDOWN:
            return self._format_to_markdown(document, personalization)
        elif format_type in [ExportFormat.PDF, ExportFormat.DOCX, ExportFormat.LATEX, ExportFormat.SLIDES_PPTX]:
            # Use pandoc for these formats
            return await self._format_with_pandoc(document, format_type, personalization, options)
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

    def _format_to_markdown_for_slides(
        self, document: GeneratedDocument, personalization: dict[str, Any] = None
    ) -> str:
        """Format document as Markdown optimized for slide generation."""
        personalization = personalization or {}

        markdown = f"# {document.title}\n\n"
        markdown += f"*{document.document_type.value.title()} - {document.detail_level.value.title()} Level*\n\n"

        # Add metadata slide
        markdown += "---\n\n"
        markdown += "## Document Information\n\n"
        markdown += f"- **Document Type:** {document.document_type.value}\n"
        markdown += f"- **Detail Level:** {document.detail_level.value}\n"
        markdown += f"- **Total Questions:** {document.total_questions}\n"
        markdown += f"- **Estimated Duration:** {document.estimated_duration} minutes\n\n"

        if document.applied_customizations:
            markdown += "- **Personalized:** Yes ✨\n\n"

        # Process sections into slides
        for section in document.sections:
            markdown += "---\n\n"  # New slide
            markdown += f"## {section.title}\n\n"

            if section.content_type == "practice_questions":
                questions = section.content_data.get("questions", [])

                # For slides, limit questions per slide
                max_questions_per_slide = 2 if personalization.get("learning_style") == "visual" else 3

                for i, question in enumerate(questions):
                    if i > 0 and i % max_questions_per_slide == 0:
                        markdown += "---\n\n"  # New slide for more questions
                        markdown += f"## {section.title} (continued)\n\n"

                    question_num = i + 1
                    markdown += f"**Question {question_num}** ({question.get('marks', 0)} marks)\n\n"
                    markdown += f"{question.get('question_text', '')}\n\n"

                    # Add visual hints for visual learners in slides
                    if personalization.get("learning_style") == "visual":
                        markdown += "> 💡 *Consider drawing a diagram to visualize this problem*\n\n"

            elif section.content_type == "learning_objectives":
                objectives_text = section.content_data.get("objectives_text", "")
                # Convert bullet points to proper slide format
                if "•" in objectives_text:
                    objectives_text = objectives_text.replace("•", "-")
                markdown += f"{objectives_text}\n\n"

            elif section.content_type == "worked_examples":
                examples = section.content_data.get("examples", [])
                for i, example in enumerate(examples):
                    if i > 0:
                        markdown += "---\n\n"  # New slide for each example
                        markdown += f"## Example {i + 1}\n\n"
                    else:
                        markdown += f"### Example {i + 1}\n\n"

                    markdown += f"**Question:** {example.get('question_text', '')}\n\n"

                    for step in example.get("solution_steps", []):
                        markdown += f"- {step.get('description_text', '')}\n"
                    markdown += "\n"

            else:
                # Generic content
                content_text = section.content_data.get("text", "")
                markdown += f"{content_text}\n\n"

        # Final slide
        markdown += "---\n\n"
        markdown += "## Thank You\n\n"
        markdown += "*Generated by Derivativ AI*\n\n"
        markdown += f"Questions? Review the {document.total_questions} problems in this {document.document_type.value}.\n\n"

        return markdown

    async def _format_with_pandoc(
        self,
        document: GeneratedDocument,
        format_type: ExportFormat,
        personalization: dict[str, Any] = None,
        options: dict[str, Any] = None,
    ) -> str:
        """Use pandoc to convert document to various formats."""
        import subprocess
        import tempfile
        from pathlib import Path

        self._think(f"Converting document to {format_type.value} using pandoc")

        personalization = personalization or {}
        options = options or {}

        try:
            # Generate markdown content (enhanced for slides if needed)
            if format_type == ExportFormat.SLIDES_PPTX:
                markdown_content = self._format_to_markdown_for_slides(document, personalization)
            else:
                markdown_content = self._format_to_markdown(document, personalization)

            # Create temporary input file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
                f.write(markdown_content)
                input_file = f.name

            # Determine output file extension
            extensions = {
                ExportFormat.PDF: '.pdf',
                ExportFormat.DOCX: '.docx',
                ExportFormat.SLIDES_PPTX: '.pptx',
                ExportFormat.LATEX: '.tex',
            }

            output_file = input_file.replace('.md', extensions[format_type])

            # Build pandoc command
            cmd = ['pandoc', input_file, '-o', output_file]

            # Add format-specific options
            if format_type == ExportFormat.PDF:
                cmd.extend([
                    '--pdf-engine=pdflatex',
                    '--variable', 'geometry:margin=2.5cm',
                    '--variable', 'fontsize=12pt',
                ])
                # Add math support
                cmd.extend(['--mathjax'])

            elif format_type == ExportFormat.DOCX:
                # Add reference doc for styling if available
                cmd.extend(['--reference-doc=/dev/null'])  # Use default styling for now

            elif format_type == ExportFormat.SLIDES_PPTX:
                cmd.extend([
                    '-t', 'pptx',
                    '--slide-level=2',  # Level 2 headers create new slides
                ])

            elif format_type == ExportFormat.LATEX:
                cmd.extend([
                    '-t', 'latex',
                    '--standalone',
                ])

            # Add metadata
            cmd.extend([
                '--metadata', f'title={document.title}',
                '--metadata', 'author=Derivativ AI',
                '--metadata', f'date={document.generated_at[:10]}',  # Just the date part
            ])

            # Add custom styling from personalization
            if personalization.get('font_size') == 'large' and format_type == ExportFormat.PDF:
                cmd.extend(['--variable', 'fontsize=14pt'])

            self._act("Executing pandoc conversion", {"command": " ".join(cmd)})

            # Execute pandoc
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Clean up input file
            Path(input_file).unlink()

            # Return the output file path
            self._act("Document converted successfully", {"output_file": output_file})
            return output_file

        except subprocess.CalledProcessError as e:
            logger.error(f"Pandoc conversion failed: {e.stderr}")
            # Clean up files
            if 'input_file' in locals():
                Path(input_file).unlink(missing_ok=True)
            if 'output_file' in locals():
                Path(output_file).unlink(missing_ok=True)
            raise ValueError(f"Document conversion failed: {e.stderr}")
        except Exception as e:
            logger.error(f"Document formatting error: {e}")
            raise ValueError(f"Document formatting failed: {e}")

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
