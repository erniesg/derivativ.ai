"""
Document Generation Service for creating educational materials.

Orchestrates the creation of worksheets, notes, textbooks, and slides
by combining question data with structured content templates.
"""

import logging
from datetime import datetime

from src.database.supabase_repository import QuestionRepository
from src.models.document_models import (
    ContentSection,
    DetailLevel,
    DocumentGenerationRequest,
    DocumentGenerationResult,
    DocumentTemplate,
    DocumentType,
    GeneratedDocument,
)
from src.models.question_models import Question
from src.services.llm_factory import LLMFactory
from src.services.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class DocumentGenerationService:
    """
    Service for generating educational documents.

    Combines question database with content generation to create
    structured educational materials at different detail levels.
    """

    def __init__(
        self,
        question_repository: QuestionRepository,
        llm_factory: LLMFactory,
        prompt_manager: PromptManager,
    ):
        self.question_repository = question_repository
        self.llm_factory = llm_factory
        self.prompt_manager = prompt_manager
        self.templates = self._load_default_templates()

    def _load_default_templates(self) -> dict[str, DocumentTemplate]:
        """Load default document templates with flexible detail levels."""
        templates = {}

        # Textbook template (supports all detail levels)
        templates["textbook_default"] = DocumentTemplate(
            name="Textbook Template",
            document_type=DocumentType.TEXTBOOK,
            supported_detail_levels=[
                DetailLevel.MINIMAL,
                DetailLevel.MEDIUM,
                DetailLevel.COMPREHENSIVE,
                DetailLevel.GUIDED,
            ],
            structure_patterns={
                DetailLevel.MINIMAL: [
                    "learning_objectives",
                    "key_concepts",
                    "essential_examples",
                    "practice_questions",
                ],
                DetailLevel.MEDIUM: [
                    "chapter_introduction",
                    "learning_objectives",
                    "concept_development",
                    "worked_examples",
                    "practice_exercises",
                    "chapter_summary",
                ],
                DetailLevel.COMPREHENSIVE: [
                    "chapter_introduction",
                    "prerequisite_knowledge",
                    "concept_development",
                    "theory_deep_dive",
                    "worked_examples",
                    "guided_practice",
                    "exercise_sets",
                    "extension_activities",
                    "real_world_applications",
                    "chapter_review",
                    "self_assessment",
                ],
                DetailLevel.GUIDED: [
                    "chapter_introduction",
                    "learning_objectives",
                    "step_by_step_guide",
                    "guided_examples",
                    "scaffolded_practice",
                    "reflection_questions",
                    "next_steps",
                ],
            },
            content_rules={
                DetailLevel.MINIMAL: {
                    "include_theory": False,
                    "max_questions_per_section": 3,
                    "focus_essential_only": True,
                },
                DetailLevel.MEDIUM: {
                    "include_theory": True,
                    "max_questions_per_section": 8,
                    "include_examples": True,
                },
                DetailLevel.COMPREHENSIVE: {
                    "include_full_theory": True,
                    "include_derivations": True,
                    "progressive_difficulty": True,
                    "include_real_world": True,
                    "include_extensions": True,
                    "max_questions_per_section": 15,
                },
                DetailLevel.GUIDED: {
                    "step_by_step_explanations": True,
                    "include_hints": True,
                    "scaffolded_difficulty": True,
                    "max_questions_per_section": 6,
                },
            },
        )

        # Notes template (supports all detail levels)
        templates["notes_default"] = DocumentTemplate(
            name="Notes Template",
            document_type=DocumentType.NOTES,
            supported_detail_levels=[
                DetailLevel.MINIMAL,
                DetailLevel.MEDIUM,
                DetailLevel.COMPREHENSIVE,
                DetailLevel.GUIDED,
            ],
            structure_patterns={
                DetailLevel.MINIMAL: ["key_points", "quick_examples", "practice_questions"],
                DetailLevel.MEDIUM: [
                    "learning_objectives",
                    "concept_explanations",
                    "worked_examples",
                    "practice_exercises",
                    "summary",
                ],
                DetailLevel.COMPREHENSIVE: [
                    "learning_objectives",
                    "detailed_theory",
                    "multiple_examples",
                    "advanced_applications",
                    "extensive_practice",
                    "comprehensive_summary",
                ],
                DetailLevel.GUIDED: [
                    "learning_objectives",
                    "key_concepts",
                    "concept_explanations",
                    "worked_examples",
                    "guided_practice",
                    "quick_check",
                    "practice_exercises",
                    "summary",
                ],
            },
            content_rules={
                DetailLevel.MINIMAL: {"bullet_points_only": True, "max_questions_per_section": 2},
                DetailLevel.MEDIUM: {
                    "include_theory": True,
                    "include_examples": True,
                    "max_questions_per_section": 5,
                },
                DetailLevel.COMPREHENSIVE: {
                    "detailed_explanations": True,
                    "multiple_perspectives": True,
                    "max_questions_per_section": 10,
                },
                DetailLevel.GUIDED: {
                    "include_theory": True,
                    "include_examples": True,
                    "include_guidance": True,
                    "structured_explanations": True,
                    "max_questions_per_section": 6,
                },
            },
        )

        # Worksheet template (supports all detail levels)
        templates["worksheet_default"] = DocumentTemplate(
            name="Worksheet Template",
            document_type=DocumentType.WORKSHEET,
            supported_detail_levels=[
                DetailLevel.MINIMAL,
                DetailLevel.MEDIUM,
                DetailLevel.COMPREHENSIVE,
                DetailLevel.GUIDED,
            ],
            structure_patterns={
                DetailLevel.MINIMAL: ["practice_questions", "answers"],
                DetailLevel.MEDIUM: [
                    "topic_overview",
                    "key_formulas",
                    "worked_examples",
                    "practice_questions",
                    "solutions",
                ],
                DetailLevel.COMPREHENSIVE: [
                    "topic_introduction",
                    "theory_review",
                    "key_formulas",
                    "worked_examples",
                    "graded_practice",
                    "challenge_questions",
                    "detailed_solutions",
                    "extension_activities",
                ],
                DetailLevel.GUIDED: [
                    "topic_overview",
                    "step_by_step_guide",
                    "guided_examples",
                    "scaffolded_practice",
                    "solutions_with_explanations",
                ],
            },
            content_rules={
                DetailLevel.MINIMAL: {"questions_only": True, "max_questions_per_section": 5},
                DetailLevel.MEDIUM: {
                    "focus_on_practice": True,
                    "include_solutions": True,
                    "minimal_theory": True,
                    "group_by_difficulty": True,
                    "max_questions_per_section": 8,
                },
                DetailLevel.COMPREHENSIVE: {
                    "include_theory": True,
                    "progressive_difficulty": True,
                    "detailed_solutions": True,
                    "max_questions_per_section": 12,
                },
                DetailLevel.GUIDED: {
                    "step_by_step_guidance": True,
                    "include_hints": True,
                    "explained_solutions": True,
                    "max_questions_per_section": 6,
                },
            },
        )

        # Slides template (supports all detail levels)
        templates["slides_default"] = DocumentTemplate(
            name="Slides Template",
            document_type=DocumentType.SLIDES,
            supported_detail_levels=[
                DetailLevel.MINIMAL,
                DetailLevel.MEDIUM,
                DetailLevel.COMPREHENSIVE,
                DetailLevel.GUIDED,
            ],
            structure_patterns={
                DetailLevel.MINIMAL: ["title_slide", "key_points", "example", "summary"],
                DetailLevel.MEDIUM: [
                    "title_slide",
                    "learning_objectives",
                    "key_concepts",
                    "examples",
                    "practice_questions",
                    "summary",
                ],
                DetailLevel.COMPREHENSIVE: [
                    "title_slide",
                    "agenda",
                    "learning_objectives",
                    "detailed_concepts",
                    "multiple_examples",
                    "applications",
                    "practice_activities",
                    "review",
                    "next_steps",
                ],
                DetailLevel.GUIDED: [
                    "title_slide",
                    "learning_objectives",
                    "step_by_step_introduction",
                    "guided_examples",
                    "interactive_practice",
                    "reflection",
                    "summary",
                ],
            },
            content_rules={
                DetailLevel.MINIMAL: {
                    "bullet_points_only": True,
                    "minimal_text": True,
                    "visual_emphasis": True,
                    "max_questions_per_section": 1,
                },
                DetailLevel.MEDIUM: {
                    "concise_content": True,
                    "visual_emphasis": True,
                    "max_questions_per_section": 2,
                },
                DetailLevel.COMPREHENSIVE: {
                    "detailed_content": True,
                    "multiple_examples": True,
                    "max_questions_per_section": 4,
                },
                DetailLevel.GUIDED: {
                    "step_by_step_approach": True,
                    "interactive_elements": True,
                    "max_questions_per_section": 3,
                },
            },
        )

        return templates

    async def generate_document(
        self, request: DocumentGenerationRequest
    ) -> DocumentGenerationResult:
        """
        Generate a complete educational document.

        Args:
            request: Document generation specification

        Returns:
            DocumentGenerationResult with generated document or error
        """
        start_time = datetime.now()

        try:
            logger.info(f"Starting document generation: {request.document_type.value}")

            # 1. Select appropriate template
            template, template_key = self._select_template(request)

            # 2. Gather relevant questions
            questions = await self._gather_questions(request)

            # 3. Generate content sections
            sections = await self._generate_sections(request, template, questions)

            # 4. Create document structure with personalization tracking
            applied_customizations = {}
            if request.custom_instructions:
                applied_customizations["custom_instructions"] = request.custom_instructions
            if request.personalization_context:
                applied_customizations["personalization_context"] = request.personalization_context

            document = GeneratedDocument(
                title=request.title,
                document_type=request.document_type,
                detail_level=request.detail_level,
                generated_at=datetime.now().isoformat(),
                template_used=template_key,
                generation_request=request,
                sections=sections,
                total_questions=len(questions),
                estimated_duration=self._estimate_duration(
                    request.document_type, request.detail_level, len(questions)
                ),
                questions_used=[q.question_id_global for q in questions],
                syllabus_coverage=request.subject_content_refs,
                applied_customizations=applied_customizations,
            )

            # 5. Generate formatted content
            await self._generate_formatted_content(document)

            processing_time = (datetime.now() - start_time).total_seconds()

            logger.info(f"Document generation completed in {processing_time:.2f}s")

            return DocumentGenerationResult(
                success=True,
                document=document,
                processing_time=processing_time,
                questions_processed=len(questions),
                sections_generated=len(sections),
                customizations_applied=len(applied_customizations),
                personalization_success=bool(
                    request.custom_instructions or request.personalization_context
                ),
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Document generation failed: {e}")

            return DocumentGenerationResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time,
            )

    def _select_template(self, request: DocumentGenerationRequest) -> tuple[DocumentTemplate, str]:
        """Select appropriate template for document generation."""
        if request.template_id and request.template_id in self.templates:
            template = self.templates[request.template_id]
            # Validate that template supports requested detail level
            if request.detail_level in template.supported_detail_levels:
                return template, request.template_id
            else:
                logger.warning(
                    f"Template {request.template_id} doesn't support {request.detail_level}, using default"
                )

        # Default template selection based on document type
        default_key = f"{request.document_type.value}_default"
        if default_key in self.templates:
            template = self.templates[default_key]
            # Check if default template supports the detail level
            if request.detail_level in template.supported_detail_levels:
                return template, default_key
            else:
                logger.warning(
                    f"Default template for {request.document_type} doesn't support {request.detail_level}"
                )

        # Fallback to first available template that supports the detail level
        for template_key, template in self.templates.items():
            if (
                template.document_type == request.document_type
                and request.detail_level in template.supported_detail_levels
            ):
                return template, template_key

        # Final fallback - return default template anyway
        fallback_template = self.templates.get(default_key, next(iter(self.templates.values())))
        fallback_key = (
            default_key if default_key in self.templates else next(iter(self.templates.keys()))
        )
        return fallback_template, fallback_key

    async def _gather_questions(self, request: DocumentGenerationRequest) -> list[Question]:
        """Gather relevant questions from database and specific references."""
        questions = []

        # 1. Add explicitly referenced questions
        for ref in request.question_references:
            question = await self.question_repository.get_question(ref.question_id)
            if question:
                questions.append(question)

        # 2. Auto-include relevant questions if requested
        if request.auto_include_questions:
            filters = {
                "tier": request.tier.value,
                "topic": request.topic,
                "limit": request.max_questions - len(questions),
                "min_quality_score": 0.7,  # Only include good quality questions
            }

            if request.grade_level:
                filters["grade_level"] = request.grade_level

            if request.subject_content_refs:
                filters["subject_content_refs"] = [
                    ref.value for ref in request.subject_content_refs
                ]

            auto_questions = await self.question_repository.list_questions(**filters)

            # Convert to Question objects (simplified - would need full conversion)
            for q_data in auto_questions[: request.max_questions - len(questions)]:
                if "content_json" in q_data:
                    question = Question(**q_data["content_json"])
                    questions.append(question)

        logger.info(f"Gathered {len(questions)} questions for document")
        return questions

    async def _generate_sections(
        self,
        request: DocumentGenerationRequest,
        template: DocumentTemplate,
        questions: list[Question],
    ) -> list[ContentSection]:
        """Generate content sections based on template and questions."""
        sections = []

        # Get structure pattern for the requested detail level
        structure_pattern = template.structure_patterns.get(
            request.detail_level,
            template.structure_patterns[DetailLevel.MEDIUM],  # fallback
        )

        for i, section_type in enumerate(structure_pattern):
            section = await self._generate_section(section_type, request, template, questions, i)
            sections.append(section)

        # Apply custom sections if specified
        if request.custom_sections:
            for j, custom_section in enumerate(request.custom_sections):
                custom_section_obj = await self._generate_section(
                    custom_section, request, template, questions, len(sections) + j
                )
                sections.append(custom_section_obj)

        return sections

    async def _generate_section(
        self,
        section_type: str,
        request: DocumentGenerationRequest,
        template: DocumentTemplate,
        questions: list[Question],
        order_index: int,
    ) -> ContentSection:
        """Generate a specific content section."""

        if section_type == "introduction":
            return self._create_introduction_section(request, order_index)
        elif section_type == "worked_examples":
            return await self._create_worked_examples_section(request, questions, order_index)
        elif section_type == "practice_questions":
            return self._create_practice_questions_section(request, questions, order_index)
        elif section_type == "solutions":
            return self._create_solutions_section(request, questions, order_index)
        elif section_type == "learning_objectives":
            return await self._create_learning_objectives_section(request, order_index)
        else:
            # Generic section generation
            return ContentSection(
                title=section_type.replace("_", " ").title(),
                content_type="generic",
                content_data={"text": f"Generated content for {section_type}"},
                order_index=order_index,
            )

    def _create_introduction_section(
        self, request: DocumentGenerationRequest, order_index: int
    ) -> ContentSection:
        """Create introduction section."""
        return ContentSection(
            title="Introduction",
            content_type="introduction",
            content_data={
                "text": f"This {request.document_type.value} covers {request.topic} "
                f"for {request.tier.value} tier students.",
                "topic": request.topic,
                "tier": request.tier.value,
                "document_type": request.document_type.value,
            },
            order_index=order_index,
        )

    async def _create_worked_examples_section(
        self, request: DocumentGenerationRequest, questions: list[Question], order_index: int
    ) -> ContentSection:
        """Create worked examples section using best questions."""
        # Select 2-3 best questions for examples
        example_questions = sorted(questions, key=lambda q: q.marks)[:3]

        examples = []
        for q in example_questions:
            example = {
                "question_text": q.raw_text_content,
                "solution_steps": q.solver_algorithm.steps if q.solver_algorithm else [],
                "marks": q.marks,
                "command_word": q.command_word.value,
            }
            examples.append(example)

        return ContentSection(
            title="Worked Examples",
            content_type="worked_examples",
            content_data={
                "examples": examples,
                "total_examples": len(examples),
            },
            order_index=order_index,
        )

    def _create_practice_questions_section(
        self, request: DocumentGenerationRequest, questions: list[Question], order_index: int
    ) -> ContentSection:
        """Create practice questions section."""
        practice_questions = []
        for q in questions:
            practice_q = {
                "question_id": q.question_id_global,
                "question_text": q.raw_text_content,
                "marks": q.marks,
                "command_word": q.command_word.value,
                "include_solution": request.include_working,
                "include_answers": request.include_answers,
            }
            practice_questions.append(practice_q)

        return ContentSection(
            title="Practice Questions",
            content_type="practice_questions",
            content_data={
                "questions": practice_questions,
                "total_marks": sum(q.marks for q in questions),
                "estimated_time": len(questions) * 3,  # 3 min per question estimate
            },
            order_index=order_index,
        )

    def _create_solutions_section(
        self, request: DocumentGenerationRequest, questions: list[Question], order_index: int
    ) -> ContentSection:
        """Create solutions section if requested."""
        if not request.include_working:
            return ContentSection(
                title="Solutions",
                content_type="solutions",
                content_data={"message": "Solutions not included in this document"},
                order_index=order_index,
            )

        solutions = []
        for q in questions:
            solution = {
                "question_id": q.question_id_global,
                "question_number": q.question_number_display,
                "solution_steps": q.solver_algorithm.steps if q.solver_algorithm else [],
                "final_answers": (
                    q.solution_and_marking_scheme.final_answers_summary
                    if q.solution_and_marking_scheme
                    else []
                ),
                "marks": q.marks,
            }
            solutions.append(solution)

        return ContentSection(
            title="Solutions",
            content_type="solutions",
            content_data={
                "solutions": solutions,
                "total_questions": len(solutions),
            },
            order_index=order_index,
        )

    async def _create_learning_objectives_section(
        self, request: DocumentGenerationRequest, order_index: int
    ) -> ContentSection:
        """Create learning objectives section using LLM."""
        # Use LLM to generate topic-appropriate learning objectives
        llm_service = self.llm_factory.create_service("openai")

        prompt = f"""
        Create 3-5 clear learning objectives for a {request.document_type.value}
        on {request.topic} for Cambridge IGCSE Mathematics {request.tier.value} tier.

        Format as a bulleted list focusing on what students will be able to do.
        """

        try:
            response = await llm_service.generate_non_stream(
                model="gpt-4o-mini",
                prompt=prompt,
                temperature=0.3,
                max_tokens=300,
            )

            return ContentSection(
                title="Learning Objectives",
                content_type="learning_objectives",
                content_data={
                    "objectives_text": response.content,
                    "topic": request.topic,
                    "tier": request.tier.value,
                },
                order_index=order_index,
            )
        except Exception as e:
            logger.warning(f"Failed to generate learning objectives: {e}")
            return ContentSection(
                title="Learning Objectives",
                content_type="learning_objectives",
                content_data={
                    "objectives_text": f"• Understand key concepts in {request.topic}\n"
                    f"• Apply {request.topic} techniques to solve problems\n"
                    f"• Work with {request.topic} in various contexts",
                    "topic": request.topic,
                    "tier": request.tier.value,
                },
                order_index=order_index,
            )

    async def _generate_formatted_content(self, document: GeneratedDocument) -> None:
        """Generate formatted content in multiple formats."""
        # Generate HTML content
        html_content = self._generate_html(document)
        document.content_html = html_content

        # Generate Markdown content
        markdown_content = self._generate_markdown(document)
        document.content_markdown = markdown_content

        # LaTeX generation would be more complex
        # document.content_latex = self._generate_latex(document)

    def _generate_html(self, document: GeneratedDocument) -> str:
        """Generate HTML formatted content."""
        html = f"""
        <html>
        <head>
            <title>{document.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin: 20px 0; }}
                .question {{ margin: 15px 0; padding: 10px; border-left: 3px solid #007acc; }}
                .solution {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>{document.title}</h1>
            <p><em>{document.document_type.value.title()} - {document.detail_level.value.title()} Level</em></p>
        """

        for section in document.sections:
            html += '<div class="section">'
            html += f"<h2>{section.title}</h2>"

            if section.content_type == "practice_questions":
                for i, q in enumerate(section.content_data.get("questions", []), 1):
                    html += f"""
                    <div class="question">
                        <strong>Question {i}</strong> ({q["marks"]} marks)<br>
                        {q["question_text"]}
                    </div>
                    """
            elif section.content_type == "solutions":
                for sol in section.content_data.get("solutions", []):
                    html += f"""
                    <div class="solution">
                        <strong>Solution {sol["question_number"]}</strong><br>
                        Steps: {len(sol.get("solution_steps", []))} steps shown<br>
                    </div>
                    """
            else:
                content_text = section.content_data.get("text", "")
                html += f"<p>{content_text}</p>"

            html += "</div>"

        html += "</body></html>"
        return html

    def _generate_markdown(self, document: GeneratedDocument) -> str:
        """Generate Markdown formatted content."""
        markdown = f"# {document.title}\n\n"
        markdown += f"*{document.document_type.value.title()} - {document.detail_level.value.title()} Level*\n\n"

        for section in document.sections:
            markdown += f"## {section.title}\n\n"

            if section.content_type == "practice_questions":
                for i, q in enumerate(section.content_data.get("questions", []), 1):
                    markdown += f"**Question {i}** ({q['marks']} marks)\n\n"
                    markdown += f"{q['question_text']}\n\n"
            elif section.content_type == "learning_objectives":
                markdown += section.content_data.get("objectives_text", "") + "\n\n"
            else:
                content_text = section.content_data.get("text", "")
                markdown += f"{content_text}\n\n"

        return markdown

    def _estimate_duration(
        self, document_type: DocumentType, detail_level: DetailLevel, question_count: int
    ) -> int:
        """Estimate completion time in minutes based on document type and detail level."""
        # Base time multipliers for document types
        base_type_multipliers = {
            DocumentType.TEXTBOOK: 1.5,
            DocumentType.NOTES: 1.2,
            DocumentType.WORKSHEET: 1.0,
            DocumentType.SLIDES: 0.5,
        }

        # Detail level multipliers
        detail_multipliers = {
            DetailLevel.MINIMAL: 1.0,
            DetailLevel.MEDIUM: 2.0,
            DetailLevel.COMPREHENSIVE: 4.0,
            DetailLevel.GUIDED: 3.0,
        }

        base_time_per_question = 2  # Base 2 minutes per question
        type_multiplier = base_type_multipliers.get(document_type, 1.0)
        detail_multiplier = detail_multipliers.get(detail_level, 2.0)

        return int(base_time_per_question * type_multiplier * detail_multiplier * question_count)

    async def get_document_templates(self) -> dict[str, DocumentTemplate]:
        """Get all available document templates."""
        return self.templates

    async def save_custom_template(self, template: DocumentTemplate) -> str:
        """Save a custom document template."""
        self.templates[template.template_id] = template
        return template.template_id

    def _apply_custom_instructions(self, content: str, custom_instructions: str) -> str:
        """Apply custom instructions to modify content."""
        # This could be expanded to use LLM for more sophisticated modifications
        if "simple language" in custom_instructions.lower():
            # In a full implementation, this would use LLM to simplify language
            pass
        if "visual learner" in custom_instructions.lower():
            # Add more visual cues, diagrams references
            content += "\n\n*Note: Refer to accompanying diagrams for visual representation.*"
        if "step-by-step" in custom_instructions.lower():
            # Ensure all procedures are broken down
            pass

        return content
