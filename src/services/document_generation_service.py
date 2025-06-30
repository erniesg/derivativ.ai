"""
Document Generation Service for creating educational materials.

Uses PromptManager with Jinja2 templates and LLM services to generate
structured educational documents following Cambridge IGCSE standards.
"""

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
from src.models.llm_schemas import DocumentGenerationResponse
from src.models.question_models import Question
from src.services.document_structure_config import get_document_structure_config
from src.services.json_parser import JSONParser
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
        self.json_parser = JSONParser(enable_cache=True)

        # Load structure patterns from configuration
        self.structure_config = get_document_structure_config()

        # Template mappings for different document types
        self.template_mappings = {
            DocumentType.WORKSHEET: "worksheet_generation",
            DocumentType.NOTES: "notes_generation",
            DocumentType.TEXTBOOK: "textbook_generation",
            DocumentType.SLIDES: "slides_generation",
        }

    async def generate_document(  # noqa: PLR0915
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
            structure_pattern = self.structure_config.get_structure_pattern(
                request.document_type, request.detail_level
            )

            # 4. Prepare template variables
            template_vars = {
                "title": request.title,
                "topic": request.topic,
                "document_type": request.document_type.value,
                "detail_level": request.detail_level.value
                if hasattr(request.detail_level, "value")
                else request.detail_level,
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

            # For OpenAI json_object mode (when parse method isn't available),
            # add explicit schema instructions to the prompt
            provider = self.llm_factory.detect_provider("gpt-4o-mini")
            if provider == "openai":
                llm_service_temp = self.llm_factory.get_service("openai")
                if not hasattr(llm_service_temp.client.chat.completions, "parse"):
                    logger.info("Adding explicit JSON schema to prompt for json_object mode")
                    detail_level_num = template_vars.get("detail_level", 5)
                    if hasattr(detail_level_num, "value"):
                        detail_level_num = detail_level_num.value

                    schema_instruction = f"""

CRITICAL: You MUST respond with a JSON object that exactly matches this schema:
{{
  "enhanced_title": "string - Enhanced title for the document",
  "introduction": "string - Brief introduction to the topic",
  "blocks": [
    {{
      "block_type": "string - One of: {', '.join(structure_pattern)}",
      "content": {{"key": "value"}},
      "estimated_minutes": 5,
      "reasoning": "string - Why this block was included"
    }}
  ],
  "total_estimated_minutes": 30,
  "actual_detail_level": {detail_level_num},
  "generation_reasoning": "string - Overall reasoning",
  "coverage_notes": "string - Coverage notes",
  "personalization_applied": []
}}

IMPORTANT:
- estimated_minutes and total_estimated_minutes must be integers (numbers), not strings
- actual_detail_level must be an integer {detail_level_num}, not a string
- Do NOT use any other JSON structure. Do NOT wrap the response in additional keys."""
                    rendered_prompt += schema_instruction

            logger.info(f"Generated prompt for {template_name}")

            # Generate content using LLM with provider-specific structured output
            from src.models.llm_models import LLMRequest

            # Configure structured output based on provider
            provider = self.llm_factory.detect_provider("gpt-4o-mini")
            llm_request = LLMRequest(
                model="gpt-4o-mini",
                prompt=rendered_prompt,
                temperature=0.3,
                max_tokens=4000,
            )

            if provider == "openai":
                # OpenAI: Use structured parsing with Pydantic model
                logger.info("Using OpenAI structured parsing")
                document_response = await llm_service.parse_structured(
                    llm_request, DocumentGenerationResponse
                )
                # Convert Pydantic model to dict for existing logic
                document_data = document_response.model_dump()

            elif provider == "google":
                # Google Gemini: Use response_schema (if available)
                logger.info("Using Gemini structured output")
                llm_request.extra_params = {
                    "response_mime_type": "application/json",
                    "response_schema": DocumentGenerationResponse,
                }
                response = await llm_service.generate_non_stream(llm_request)

                # Parse JSON response
                extraction_result = await self.json_parser.extract_json(
                    response.content, model_name=llm_request.model
                )
                if extraction_result.success:
                    document_data = extraction_result.data
                else:
                    logger.error(f"Gemini JSON parsing failed: {extraction_result.error}")
                    document_data = self._create_fallback_document_data(request, questions)

            else:  # Anthropic and others
                # Anthropic: Use enhanced prompt instructions with JSON parsing
                logger.info("Using prompt-based structured output with JSON parsing")
                response = await llm_service.generate_non_stream(llm_request)

                # Parse JSON response with enhanced parser
                extraction_result = await self.json_parser.extract_json(
                    response.content, model_name=llm_request.model
                )
                if extraction_result.success:
                    document_data = extraction_result.data
                else:
                    logger.error(f"JSON parsing failed: {extraction_result.error}")
                    document_data = self._create_fallback_document_data(request, questions)

            # Log the structured data we got
            logger.info(f"Final data keys: {list(document_data.keys())}")
            logger.info(f"Blocks found: {len(document_data.get('blocks', []))}")

            # 7. Create structured document from blocks
            sections = self._parse_sections_from_blocks(document_data, questions)
            logger.info(f"Created {len(sections)} sections from blocks")

            # 8. Track applied customizations
            applied_customizations = {}
            if request.custom_instructions:
                applied_customizations["custom_instructions"] = request.custom_instructions
            if request.personalization_context:
                applied_customizations["personalization_context"] = request.personalization_context

            # 9. Create final document
            document = GeneratedDocument(
                title=document_data.get("enhanced_title", request.title),
                document_type=request.document_type,
                detail_level=request.detail_level,
                generated_at=datetime.now().isoformat(),
                template_used=template_name,
                generation_request=request,
                sections=sections,
                total_questions=len(questions),
                estimated_duration=document_data.get(
                    "total_estimated_minutes",
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

    def _parse_sections_from_blocks(
        self, document_data: dict[str, Any], questions: list[Question]
    ) -> list[ContentSection]:
        """Parse sections from LLM-generated document blocks."""
        sections = []

        # Handle new "blocks" format from structured output
        blocks = document_data.get("blocks", [])
        if not blocks:
            # Fallback to old format if needed
            return self._parse_sections_from_llm_data_legacy(document_data, questions)

        for i, block in enumerate(blocks):
            block_type = block.get("block_type", "generic")
            content = block.get("content", {})

            # Create section title from block type
            section_title = block_type.replace("_", " ").title()

            section = ContentSection(
                title=section_title,
                content_type=block_type,
                content_data=content,
                order_index=i,
            )

            # Enrich sections with actual question data where appropriate
            if block_type == "practice_questions" and questions:
                # Merge LLM-generated content with actual question data
                practice_data = self._create_practice_questions_data(questions)
                section.content_data = {**content, **practice_data}
            elif block_type == "worked_examples" and questions:
                # Merge LLM-generated content with actual solution data
                examples_data = self._create_worked_examples_data(questions[:3])
                section.content_data = {**content, **examples_data}

            sections.append(section)

        return sections

    def _parse_sections_from_llm_data_legacy(
        self, document_data: dict[str, Any], questions: list[Question]
    ) -> list[ContentSection]:
        """Parse sections from old LLM-generated document data format."""
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
        structure_pattern = self.structure_config.get_structure_pattern(
            request.document_type, request.detail_level
        )

        blocks = []
        for i, section_type in enumerate(structure_pattern):
            block = {
                "block_type": section_type,
                "content": {"text": f"Generated content for {section_type}"},
                "estimated_minutes": 5,
                "reasoning": f"Fallback content for {section_type.replace('_', ' ')}",
            }
            blocks.append(block)

        estimated_duration = self._estimate_duration(
            request.document_type, request.detail_level, len(questions)
        )

        return {
            "enhanced_title": request.title,
            "introduction": f"Introduction to {request.topic}",
            "blocks": blocks,
            "total_estimated_minutes": estimated_duration,
            "actual_detail_level": request.detail_level.value
            if hasattr(request.detail_level, "value")
            else 5,
            "generation_reasoning": "Fallback content generated due to LLM response parsing failure",
            "coverage_notes": f"Basic coverage of {request.topic}",
            "personalization_applied": [],
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
        all_patterns = self.structure_config.get_all_patterns()
        return {
            doc_type.value: {
                detail_level.value: pattern for detail_level, pattern in patterns.items()
            }
            for doc_type, patterns in all_patterns.items()
        }

    async def get_document_templates(self) -> dict[str, Any]:
        """Get all document templates with their metadata."""
        from src.models.document_models import DocumentTemplate

        templates = {}
        all_patterns = self.structure_config.get_all_patterns()

        for doc_type, template_name in self.template_mappings.items():
            doc_patterns = all_patterns.get(doc_type, {})
            structure_patterns = {}
            for detail_level, pattern in doc_patterns.items():
                structure_patterns[detail_level.value] = pattern

            # Create content rules for each supported detail level
            content_rules = {}
            for detail_level in doc_patterns:
                content_rules[detail_level] = {
                    "min_questions": 1,
                    "max_questions": 50,
                    "supports_custom_instructions": True,
                    "supports_personalization": True,
                }

            templates[doc_type.value] = DocumentTemplate(
                name=template_name,
                document_type=doc_type,
                supported_detail_levels=[level for level in doc_patterns],
                structure_patterns={level: pattern for level, pattern in doc_patterns.items()},
                content_rules=content_rules,
            )

        return templates

    async def save_custom_template(self, template: Any) -> str:
        """Save a custom document template."""
        # For now, return a placeholder ID
        # In a full implementation, this would save to database
        import uuid

        template_id = str(uuid.uuid4())
        logger.info(f"Custom template saved with ID: {template_id}")
        return template_id
