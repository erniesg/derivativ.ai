"""
Document Generation Service for creating educational materials.

Uses PromptManager with Jinja2 templates and LLM services to generate
structured educational documents following Cambridge IGCSE standards.
"""

import json
import logging
from datetime import datetime
from typing import Any

from src.database.supabase_repository import QuestionRepository
from src.models.document_models import (
    ContentSection,
    DetailLevel,
    DocumentGenerationRequest,
    DocumentGenerationResult,
    DocumentType,
    GeneratedDocument,
)
from src.models.question_models import Question
from src.services.llm_factory import LLMFactory
from src.services.prompt_manager import PromptConfig, PromptManager

logger = logging.getLogger(__name__)


class DocumentGenerationService:
    """
    Service for generating educational documents using LLM and PromptManager.

    This service follows the proper architecture:
    1. Uses PromptManager with Jinja2 templates
    2. LLM generates document content first
    3. Content is then structured and processed
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

        # Template mappings for different document types
        self.template_mappings = {
            DocumentType.WORKSHEET: "worksheet_generation",
            DocumentType.NOTES: "notes_generation",
            DocumentType.TEXTBOOK: "textbook_generation",
            DocumentType.SLIDES: "slides_generation",
        }

        # Structure patterns for each document type and detail level
        self.structure_patterns = {
            DocumentType.WORKSHEET: {
                DetailLevel.MINIMAL: ["practice_questions", "answers"],
                DetailLevel.MEDIUM: [
                    "learning_objectives",
                    "key_formulas",
                    "worked_examples",
                    "practice_questions",
                    "solutions",
                ],
                DetailLevel.COMPREHENSIVE: [
                    "learning_objectives",
                    "topic_introduction",
                    "key_concepts",
                    "worked_examples",
                    "graded_practice",
                    "challenge_questions",
                    "detailed_solutions",
                ],
                DetailLevel.GUIDED: [
                    "learning_objectives",
                    "step_by_step_guide",
                    "guided_examples",
                    "scaffolded_practice",
                    "solutions_with_explanations",
                ],
            },
            DocumentType.NOTES: {
                DetailLevel.MINIMAL: ["key_concepts", "quick_examples", "practice_questions"],
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
                    "step_by_step_concepts",
                    "guided_examples",
                    "scaffolded_practice",
                    "self_check",
                ],
            },
            DocumentType.TEXTBOOK: {
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
                    "detailed_theory",
                    "worked_examples",
                    "applications",
                    "extensive_exercises",
                    "extension_activities",
                    "chapter_review",
                ],
                DetailLevel.GUIDED: [
                    "learning_pathway",
                    "step_by_step_development",
                    "guided_discovery",
                    "scaffolded_practice",
                    "reflection_activities",
                ],
            },
            DocumentType.SLIDES: {
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
                    "summary",
                ],
                DetailLevel.GUIDED: [
                    "title_slide",
                    "learning_objectives",
                    "step_by_step_introduction",
                    "guided_examples",
                    "interactive_practice",
                    "reflection",
                ],
            },
        }

    async def generate_document(
        self, request: DocumentGenerationRequest
    ) -> DocumentGenerationResult:
        """
        Generate a complete educational document using LLM and templates.

        Args:
            request: Document generation specification

        Returns:
            DocumentGenerationResult with generated document or error
        """
        start_time = datetime.now()

        try:
            logger.info(f"Starting document generation: {request.document_type.value}")

            # 1. Gather relevant questions
            questions = await self._gather_questions(request)

            # 2. Get template name for document type
            template_name = self.template_mappings.get(request.document_type, "document_generation")

            # 3. Get structure pattern for detail level
            structure_pattern = self.structure_patterns[request.document_type][request.detail_level]

            # 4. Prepare template variables
            template_vars = {
                "title": request.title,
                "topic": request.topic,
                "document_type": request.document_type.value,
                "detail_level": request.detail_level.value,
                "target_grade": request.grade_level or 7,
                "structure_pattern": structure_pattern,
            }

            # Add optional variables
            if request.tier:
                template_vars["tier"] = request.tier.value
            if request.custom_instructions:
                template_vars["custom_instructions"] = request.custom_instructions
            if request.personalization_context:
                template_vars["personalization_context"] = request.personalization_context

            # 5. Generate document content using LLM
            llm_service = self.llm_factory.get_service("openai")

            prompt_config = PromptConfig(
                template_name=template_name,
                variables=template_vars,
            )

            rendered_prompt = await self.prompt_manager.render_prompt(
                prompt_config, model_name="gpt-4o-mini"
            )

            logger.info(f"Generated prompt for {template_name}")

            # Generate content using LLM
            response = await llm_service.generate_non_stream(
                model="gpt-4o-mini",
                prompt=rendered_prompt,
                temperature=0.3,
                max_tokens=4000,
            )

            # 6. Parse LLM response
            try:
                document_data = json.loads(response.content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                # Fallback: create basic document structure
                document_data = self._create_fallback_document_data(request, questions)

            # 7. Create structured document
            sections = self._parse_sections_from_llm_data(document_data, questions)

            # 8. Track applied customizations
            applied_customizations = {}
            if request.custom_instructions:
                applied_customizations["custom_instructions"] = request.custom_instructions
            if request.personalization_context:
                applied_customizations["personalization_context"] = request.personalization_context

            # 9. Create final document
            document = GeneratedDocument(
                title=document_data.get("title", request.title),
                document_type=request.document_type,
                detail_level=request.detail_level,
                generated_at=datetime.now().isoformat(),
                template_used=template_name,
                generation_request=request,
                sections=sections,
                total_questions=document_data.get("total_questions", len(questions)),
                estimated_duration=document_data.get(
                    "estimated_duration",
                    self._estimate_duration(
                        request.document_type, request.detail_level, len(questions)
                    ),
                ),
                questions_used=[q.question_id_global for q in questions],
                syllabus_coverage=request.subject_content_refs,
                applied_customizations=applied_customizations,
            )

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
                "tier": request.tier.value if request.tier else "core",
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

            try:
                auto_questions = await self.question_repository.list_questions(**filters)

                # Convert to Question objects (simplified - would need full conversion)
                for q_data in auto_questions[: request.max_questions - len(questions)]:
                    if "content_json" in q_data:
                        question = Question(**q_data["content_json"])
                        questions.append(question)
            except Exception as e:
                logger.warning(f"Failed to auto-gather questions: {e}")

        logger.info(f"Gathered {len(questions)} questions for document")
        return questions

    def _parse_sections_from_llm_data(
        self, document_data: dict[str, Any], questions: list[Question]
    ) -> list[ContentSection]:
        """Parse sections from LLM-generated document data."""
        sections = []

        for i, section_data in enumerate(document_data.get("sections", [])):
            section = ContentSection(
                title=section_data.get("title", f"Section {i+1}"),
                content_type=section_data.get("content_type", "generic"),
                content_data=section_data.get("content_data", {}),
                order_index=section_data.get("order_index", i),
            )

            # Enrich sections with actual question data where appropriate
            if section.content_type == "practice_questions" and questions:
                section.content_data = self._create_practice_questions_data(questions)
            elif section.content_type == "worked_examples" and questions:
                section.content_data = self._create_worked_examples_data(questions[:3])

            sections.append(section)

        return sections

    def _create_practice_questions_data(self, questions: list[Question]) -> dict[str, Any]:
        """Create practice questions data from Question objects."""
        practice_questions = []
        for q in questions:
            practice_q = {
                "question_id": q.question_id_global,
                "question_text": q.raw_text_content,
                "marks": q.marks,
                "command_word": q.command_word.value,
            }
            practice_questions.append(practice_q)

        return {
            "questions": practice_questions,
            "total_marks": sum(q.marks for q in questions),
            "estimated_time": len(questions) * 3,  # 3 min per question estimate
        }

    def _create_worked_examples_data(self, questions: list[Question]) -> dict[str, Any]:
        """Create worked examples data from Question objects."""
        examples = []
        for q in questions:
            example = {
                "question_text": q.raw_text_content,
                "marks": q.marks,
                "command_word": q.command_word.value,
                "solution_steps": (
                    [
                        {"step_number": i + 1, "description_text": step}
                        for i, step in enumerate(q.solver_algorithm.steps)
                    ]
                    if q.solver_algorithm
                    else []
                ),
            }
            examples.append(example)

        return {
            "examples": examples,
            "total_examples": len(examples),
        }

    def _create_fallback_document_data(
        self, request: DocumentGenerationRequest, questions: list[Question]
    ) -> dict[str, Any]:
        """Create fallback document data when LLM response parsing fails."""
        structure_pattern = self.structure_patterns[request.document_type][request.detail_level]

        sections = []
        for i, section_type in enumerate(structure_pattern):
            section = {
                "title": section_type.replace("_", " ").title(),
                "content_type": section_type,
                "content_data": {"text": f"Generated content for {section_type}"},
                "order_index": i,
            }
            sections.append(section)

        return {
            "title": request.title,
            "document_type": request.document_type.value,
            "detail_level": request.detail_level.value,
            "sections": sections,
            "estimated_duration": self._estimate_duration(
                request.document_type, request.detail_level, len(questions)
            ),
            "total_questions": len(questions),
        }

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

        estimated_time = int(
            base_time_per_question * type_multiplier * detail_multiplier * question_count
        )

        # Minimum time based on detail level
        min_times = {
            DetailLevel.MINIMAL: 5,
            DetailLevel.MEDIUM: 15,
            DetailLevel.COMPREHENSIVE: 30,
            DetailLevel.GUIDED: 20,
        }

        return max(estimated_time, min_times.get(detail_level, 15))

    async def get_available_templates(self) -> dict[str, str]:
        """Get available document generation templates."""
        return {
            doc_type.value: template_name
            for doc_type, template_name in self.template_mappings.items()
        }

    async def get_structure_patterns(self) -> dict[str, dict[str, list[str]]]:
        """Get structure patterns for all document types and detail levels."""
        return {
            doc_type.value: {
                detail_level.value: pattern for detail_level, pattern in patterns.items()
            }
            for doc_type, patterns in self.structure_patterns.items()
        }
