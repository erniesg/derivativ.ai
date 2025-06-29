"""
Document Generation Service V2 with blocks-based architecture.

Uses the new content blocks system with LLM-driven content generation
and flexible time/detail constraints.
"""

import json
import logging
from datetime import datetime
from typing import Any

from src.models.content_blocks import get_block_class
from src.models.document_blueprints import BlockConfig, get_blueprint
from src.models.document_generation_v2 import (
    BLOCK_CONTENT_SCHEMAS,
    DOCUMENT_CONTENT_SCHEMA,
    BlockGenerationResult,
    BlockSelectionResult,
    DocumentContentStructure,
    DocumentGenerationRequestV2,
    DocumentGenerationResultV2,
    GeneratedDocumentV2,
    GenerationApproach,
    SelectedBlock,
)
from src.models.enums import get_detailed_syllabus_content
from src.services.llm_factory import LLMFactory
from src.services.prompt_manager import PromptConfig, PromptManager

logger = logging.getLogger(__name__)


class BlockSelector:
    """Handles selection of content blocks based on constraints."""

    def __init__(self, llm_factory: LLMFactory, prompt_manager: PromptManager):
        self.llm_factory = llm_factory
        self.prompt_manager = prompt_manager

    async def select_blocks(self, request: DocumentGenerationRequestV2) -> BlockSelectionResult:
        """
        Select blocks for inclusion based on request constraints.

        Args:
            request: Document generation request

        Returns:
            BlockSelectionResult with selected blocks and reasoning
        """
        blueprint = get_blueprint(request.document_type)
        detail_level = request.get_effective_detail_level()

        if request.generation_approach == GenerationApproach.RULE_BASED:
            return await self._rule_based_selection(blueprint, request, detail_level)
        elif request.generation_approach == GenerationApproach.LLM_DRIVEN:
            return await self._llm_driven_selection(blueprint, request, detail_level)
        else:  # HYBRID
            return await self._hybrid_selection(blueprint, request, detail_level)

    async def _rule_based_selection(
        self, blueprint, request: DocumentGenerationRequestV2, detail_level: int
    ) -> BlockSelectionResult:
        """Select blocks using rule-based logic."""
        # Get applicable blocks at this detail level
        applicable_blocks = blueprint.get_applicable_blocks(detail_level)

        # Start with required blocks
        selected = [
            SelectedBlock(
                block_config=block,
                content_guidelines=self._get_content_guidelines(block, request),
                estimated_content_volume=self._estimate_content_volume(block, request),
            )
            for block in blueprint.get_required_blocks()
            if block.is_applicable(detail_level)
        ]

        # Add high priority blocks if time allows
        remaining_time = request.target_duration_minutes or 30
        current_time = blueprint.estimate_time(
            [s.block_config for s in selected],
            self._get_total_content_volume([s.block_config for s in selected], request),
        )

        # Add blocks by priority if we have time
        for block in applicable_blocks:
            if block not in [s.block_config for s in selected] and block.priority.value in [
                "high",
                "medium",
            ]:
                # Estimate time if we add this block
                test_blocks = [s.block_config for s in selected] + [block]
                test_time = blueprint.estimate_time(
                    test_blocks, self._get_total_content_volume(test_blocks, request)
                )

                if test_time <= remaining_time * 1.1:  # 10% buffer
                    selected.append(
                        SelectedBlock(
                            block_config=block,
                            content_guidelines=self._get_content_guidelines(block, request),
                            estimated_content_volume=self._estimate_content_volume(block, request),
                        )
                    )
                    current_time = test_time

        # Apply user overrides
        selected = self._apply_user_overrides(selected, request, blueprint)

        final_time = blueprint.estimate_time(
            [s.block_config for s in selected],
            self._get_total_content_volume([s.block_config for s in selected], request),
        )

        return BlockSelectionResult(
            selected_blocks=selected,
            total_estimated_minutes=final_time,
            selection_reasoning=f"Rule-based selection for {request.document_type} at detail level {detail_level}. "
            f"Selected {len(selected)} blocks targeting {remaining_time} minutes.",
            excluded_blocks=[
                f"{block.block_type} (priority: {block.priority}, min_detail: {block.min_detail_level})"
                for block in applicable_blocks
                if block not in [s.block_config for s in selected]
            ],
        )

    async def _llm_driven_selection(
        self, blueprint, request: DocumentGenerationRequestV2, detail_level: int
    ) -> BlockSelectionResult:
        """Let LLM decide block selection based on constraints."""
        # Get all available blocks
        applicable_blocks = blueprint.get_applicable_blocks(detail_level)

        # Create context for LLM
        context = {
            "document_type": request.document_type.value,
            "topic": request.topic.value,  # Use enum value
            "target_duration_minutes": request.target_duration_minutes,
            "detail_level": detail_level,
            "target_grade": request.grade_level or 8,  # Add missing required variable
            "tier": request.tier.value if request.tier else "Core",  # Add tier
            "available_blocks": [
                {
                    "block_type": block.block_type,
                    "priority": block.priority.value,
                    "estimated_minutes": get_block_class(block.block_type)().estimated_minutes,
                    "customization_hints": block.customization_hints,
                }
                for block in applicable_blocks
            ],
            "required_blocks": [
                block.block_type
                for block in blueprint.get_required_blocks()
                if block.is_applicable(detail_level)
            ],
            "custom_instructions": request.custom_instructions,
            "force_include": request.force_include_blocks,
            "exclude": request.exclude_blocks,
        }

        # Generate selection using LLM
        llm_service = self.llm_factory.get_service("openai")

        prompt_config = PromptConfig(template_name="block_selection", variables=context)

        rendered_prompt = await self.prompt_manager.render_prompt(
            prompt_config, model_name="gpt-4o-mini"
        )

        # For now, fall back to rule-based if template doesn't exist
        logger.warning("LLM-driven selection not fully implemented, using rule-based fallback")
        return await self._rule_based_selection(blueprint, request, detail_level)

    async def _hybrid_selection(
        self, blueprint, request: DocumentGenerationRequestV2, detail_level: int
    ) -> BlockSelectionResult:
        """Hybrid approach: rules select, LLM can adjust."""
        # Start with rule-based selection
        rule_result = await self._rule_based_selection(blueprint, request, detail_level)

        # TODO: Allow LLM to make adjustments based on context
        # For now, just return rule-based result
        return rule_result

    def _get_content_guidelines(
        self, block: BlockConfig, request: DocumentGenerationRequestV2
    ) -> dict[str, Any]:
        """Get content guidelines for a block."""
        guidelines = {
            "topic": request.topic,
            "subtopics": request.subtopics,
            "grade_level": request.grade_level,
            "difficulty": request.difficulty,
            "tier": request.tier.value,
            "custom_instructions": request.custom_instructions,
        }

        # Add block-specific customization hints
        guidelines.update(block.customization_hints)

        return guidelines

    def _estimate_content_volume(
        self, block: BlockConfig, request: DocumentGenerationRequestV2
    ) -> dict[str, int]:
        """Estimate content volume for a block."""
        volume = {}

        if block.block_type == "practice_questions":
            volume["num_questions"] = request.num_questions or 5
        elif block.block_type == "worked_example":
            # Extract from customization hints or use defaults
            num_examples = block.customization_hints.get("num_examples", "2")
            if isinstance(num_examples, str) and "-" in num_examples:
                # Handle ranges like "2-4"
                start, end = map(int, num_examples.split("-"))
                volume["num_examples"] = (start + end) // 2  # Use middle value
            else:
                volume["num_examples"] = 2

        return volume

    def _get_total_content_volume(
        self, blocks: list[BlockConfig], request: DocumentGenerationRequestV2
    ) -> dict[str, int]:
        """Get total content volume across all blocks."""
        total_volume = {"num_questions": 0, "num_examples": 0}

        for block in blocks:
            volume = self._estimate_content_volume(block, request)
            for key, value in volume.items():
                total_volume[key] += value

        return total_volume

    def _apply_user_overrides(
        self, selected: list[SelectedBlock], request: DocumentGenerationRequestV2, blueprint
    ) -> list[SelectedBlock]:
        """Apply user's block inclusion/exclusion preferences."""
        # Remove excluded blocks
        if request.exclude_blocks:
            selected = [
                s for s in selected if s.block_config.block_type not in request.exclude_blocks
            ]

        # Add force-included blocks
        if request.force_include_blocks:
            existing_types = {s.block_config.block_type for s in selected}

            for block_type in request.force_include_blocks:
                if block_type not in existing_types:
                    # Find the block config
                    for block in blueprint.blocks:
                        if block.block_type == block_type:
                            selected.append(
                                SelectedBlock(
                                    block_config=block,
                                    content_guidelines=self._get_content_guidelines(block, request),
                                    estimated_content_volume=self._estimate_content_volume(
                                        block, request
                                    ),
                                )
                            )
                            break

        return selected


class DocumentGenerationServiceV2:
    """
    Enhanced document generation service using blocks-based architecture.

    Provides flexible document generation with LLM-driven content creation
    and constraint-based block selection.
    """

    def __init__(
        self,
        llm_factory: LLMFactory,
        prompt_manager: PromptManager,
        question_generation_service=None,
        question_repository=None,
        document_storage_repository=None,
    ):
        self.llm_factory = llm_factory
        self.prompt_manager = prompt_manager
        self.question_generation_service = question_generation_service
        self.question_repository = question_repository
        self.document_storage_repository = document_storage_repository
        self.block_selector = BlockSelector(llm_factory, prompt_manager)

    async def generate_document(
        self, request: DocumentGenerationRequestV2
    ) -> DocumentGenerationResultV2:
        """
        Generate a document using the blocks-based approach.

        Args:
            request: Document generation request

        Returns:
            DocumentGenerationResultV2 with generated document or error
        """
        start_time = datetime.now()

        try:
            logger.info(f"Starting document generation v2: {request.document_type.value}")

            # 1. Select blocks based on constraints
            selection_result = await self.block_selector.select_blocks(request)

            logger.info(
                f"Selected {len(selection_result.selected_blocks)} blocks, "
                f"estimated time: {selection_result.total_estimated_minutes} minutes"
            )

            # 2. Generate content using LLM with structured output
            content_structure = await self._generate_content_structure(request, selection_result)

            # 3. Render blocks to final content
            rendered_blocks = await self._render_blocks(content_structure)

            # 4. Create final document
            blueprint = get_blueprint(request.document_type)

            document = GeneratedDocumentV2(
                title=content_structure.enhanced_title or request.title,
                document_type=request.document_type,
                generation_request=request,
                blueprint_used=blueprint.name,
                content_structure=content_structure,
                rendered_blocks=rendered_blocks,
                total_estimated_minutes=content_structure.total_estimated_minutes,
                actual_detail_level=content_structure.actual_detail_level,
                word_count=self._calculate_word_count(rendered_blocks),
                available_formats=blueprint.supported_formats,
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            logger.info(f"Document generation completed in {processing_time:.2f}s")

            # Save document to storage if repository is available
            document_id = await self._save_document_to_storage(request, document)

            return DocumentGenerationResultV2(
                success=True,
                document=document,
                processing_time=processing_time,
                generation_insights={
                    "blocks_selected": len(selection_result.selected_blocks),
                    "selection_reasoning": selection_result.selection_reasoning,
                    "llm_reasoning": content_structure.generation_reasoning,
                    "document_id": document_id,
                },
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Document generation failed: {e}")

            # Save failed document metadata
            document_id = await self._save_failed_document_to_storage(request, str(e))

            return DocumentGenerationResultV2(
                success=False,
                error_message=str(e),
                processing_time=processing_time,
                generation_insights={
                    "document_id": document_id,
                    "error": str(e),
                },
            )

    async def _generate_content_structure(
        self, request: DocumentGenerationRequestV2, selection_result: BlockSelectionResult
    ) -> DocumentContentStructure:
        """Generate content structure using LLM with structured output."""
        # Prepare context for LLM generation
        context = {
            "document_type": request.document_type.value,
            "title": request.title,
            "topic": request.topic.value,  # Use enum value
            "subtopics": request.subtopics,
            "detail_level": request.get_effective_detail_level(),
            "target_duration_minutes": request.target_duration_minutes,
            "grade_level": request.grade_level,
            "difficulty": request.difficulty,
            "tier": request.tier.value,
            "custom_instructions": request.custom_instructions,
            "personalization_context": request.personalization_context,
            "syllabus_refs": request.get_syllabus_refs(),  # Add syllabus references
            "detailed_syllabus_content": get_detailed_syllabus_content(
                request.get_syllabus_refs()
            ),  # Detailed content
            # Selected blocks with their guidelines
            "selected_blocks": [
                {
                    "block_type": block.block_config.block_type,
                    "content_guidelines": block.content_guidelines,
                    "estimated_content_volume": block.estimated_content_volume,
                    "schema": BLOCK_CONTENT_SCHEMAS.get(block.block_config.block_type, {}),
                }
                for block in selection_result.selected_blocks
            ],
            # Output structure requirements
            "output_schema": DOCUMENT_CONTENT_SCHEMA,
        }

        # Generate content using LLM
        try:
            llm_service = self.llm_factory.get_service("openai")
            logger.debug("✅ LLM service obtained successfully")

            prompt_config = PromptConfig(
                template_name="document_content_generation", variables=context
            )
            logger.debug("✅ Prompt config created")

            rendered_prompt = await self.prompt_manager.render_prompt(
                prompt_config, model_name="gpt-4o-mini"
            )
            logger.debug(f"✅ Prompt rendered successfully ({len(rendered_prompt)} chars)")
        except Exception as e:
            logger.error(f"❌ Error in prompt setup: {e}")
            raise e

        from src.models.llm_models import LLMRequest

        llm_request = LLMRequest(
            model="gpt-4o-mini",
            prompt=rendered_prompt,
            temperature=0.3,
            max_tokens=8000,
            response_format={"type": "json_object"},  # Request JSON output
        )

        try:
            logger.debug(f"📤 Sending LLM request: {llm_request.model}")
            response = await llm_service.generate_non_stream(llm_request)
            logger.debug(f"✅ LLM response received ({len(response.content)} chars)")
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            raise e

        # Parse structured output
        try:
            # Handle case where LLM wraps JSON in markdown code blocks
            content_to_parse = response.content
            if content_to_parse.startswith("```json"):
                # Extract JSON from markdown code block
                content_to_parse = content_to_parse[7:]  # Remove ```json
                if content_to_parse.endswith("```"):
                    content_to_parse = content_to_parse[:-3]  # Remove closing ```
                content_to_parse = content_to_parse.strip()
            elif content_to_parse.startswith("```"):
                # Extract from generic code block
                content_to_parse = content_to_parse[3:]
                if content_to_parse.endswith("```"):
                    content_to_parse = content_to_parse[:-3]
                content_to_parse = content_to_parse.strip()

            content_data = json.loads(content_to_parse)

            # Handle case where LLM nests under "properties" (common with JSON schema)
            if "properties" in content_data and "blocks" not in content_data:
                logger.warning("LLM nested response under 'properties', extracting...")
                content_data = content_data["properties"]

            # Validate against schema (basic validation)
            self._validate_content_structure(content_data)

            # Convert to DocumentContentStructure
            blocks = [
                BlockGenerationResult(**block_data) for block_data in content_data.get("blocks", [])
            ]

            content_structure = DocumentContentStructure(
                enhanced_title=content_data.get("enhanced_title"),
                introduction=content_data.get("introduction"),
                blocks=blocks,
                total_estimated_minutes=content_data["total_estimated_minutes"],
                actual_detail_level=content_data["actual_detail_level"],
                generation_reasoning=content_data["generation_reasoning"],
                coverage_notes=content_data.get("coverage_notes"),
                personalization_applied=content_data.get("personalization_applied", []),
            )

            # Generate questions if needed and service available
            await self._generate_questions_if_needed(content_structure, request)

            return content_structure

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse structured output: {e}")
            logger.error(f"Raw LLM response: {response.content[:500]}...")  # Show first 500 chars
            # Fallback to creating basic structure
            return self._create_fallback_structure(request, selection_result)
        except Exception as e:
            logger.error(f"Unexpected error in content generation: {type(e).__name__}: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback to creating basic structure
            return self._create_fallback_structure(request, selection_result)

    async def _render_blocks(self, content_structure: DocumentContentStructure) -> dict[str, str]:
        """Render each block to its final content format."""
        rendered = {}

        for block_result in content_structure.blocks:
            try:
                block_class = get_block_class(block_result.block_type)
                block_instance = block_class()

                rendered_content = await block_instance.render(block_result.content)

                rendered[block_result.block_type] = rendered_content

            except Exception as e:
                logger.error(f"Failed to render block {block_result.block_type}: {e}")
                rendered[
                    block_result.block_type
                ] = f"<!-- Error rendering {block_result.block_type}: {e} -->"

        return rendered

    def _validate_content_structure(self, content_data: dict[str, Any]):
        """Basic validation of structured output against schema."""
        required_fields = DOCUMENT_CONTENT_SCHEMA["required"]

        for field in required_fields:
            if field not in content_data:
                raise ValueError(f"Missing required field: {field}")

        # Validate blocks have required fields
        for block in content_data.get("blocks", []):
            if "block_type" not in block:
                raise ValueError("Block missing block_type")
            if "content" not in block:
                raise ValueError("Block missing content")
            if "estimated_minutes" not in block:
                raise ValueError("Block missing estimated_minutes")

    def _create_fallback_structure(
        self, request: DocumentGenerationRequestV2, selection_result: BlockSelectionResult
    ) -> DocumentContentStructure:
        """Create fallback structure when LLM generation fails."""
        blocks = []

        for selected_block in selection_result.selected_blocks:
            # Create minimal content for each block
            fallback_content = self._create_fallback_block_content(
                selected_block.block_config.block_type, request
            )

            blocks.append(
                BlockGenerationResult(
                    block_type=selected_block.block_config.block_type,
                    content=fallback_content,
                    estimated_minutes=get_block_class(
                        selected_block.block_config.block_type
                    )().estimated_minutes,
                    reasoning="Fallback content due to generation error",
                )
            )

        return DocumentContentStructure(
            enhanced_title=request.title,
            blocks=blocks,
            total_estimated_minutes=selection_result.total_estimated_minutes,
            actual_detail_level=request.get_effective_detail_level(),
            generation_reasoning="Fallback structure created due to LLM generation failure",
        )

    def _create_fallback_block_content(
        self, block_type: str, request: DocumentGenerationRequestV2
    ) -> dict[str, Any]:
        """Create minimal fallback content for a block type."""
        content_map = {
            "learning_objectives": {
                "objectives": [
                    f"Understand {request.topic}",
                    f"Apply key concepts from {request.topic}",
                ]
            },
            "concept_explanation": {
                "concepts": [
                    {
                        "name": f"{request.topic} Fundamentals",
                        "explanation": f"Key concepts and principles of {request.topic}",
                    }
                ]
            },
            "practice_questions": {
                "questions": [
                    {
                        "text": f"Solve a basic {request.topic} problem",
                        "marks": 3,
                        "difficulty": "medium",
                    }
                ]
            },
            "summary": {
                "key_points": [
                    f"Reviewed fundamental concepts of {request.topic}",
                    "Practiced key problem-solving techniques",
                ]
            },
        }

        return content_map.get(
            block_type, {"content": f"Content for {block_type} related to {request.topic}"}
        )

    async def _generate_questions_if_needed(
        self, content_structure: DocumentContentStructure, request: DocumentGenerationRequestV2
    ):
        """Get questions from DB first, then generate if needed."""
        # Find practice question blocks that need actual questions
        for block in content_structure.blocks:
            if block.block_type == "practice_questions":
                questions_data = block.content.get("questions", [])
                num_needed = len(questions_data)

                if not questions_data or not request.include_questions:
                    continue

                real_questions = []

                # 1. Try to get questions from database first
                if self.question_repository:
                    try:
                        # Create topic keywords from request
                        topic_keywords = [request.topic.value.lower()]
                        if request.subtopics:
                            topic_keywords.extend([st.lower() for st in request.subtopics])

                        db_questions = self.question_repository.search_questions_by_content(
                            subject_content_refs=request.get_syllabus_refs(),
                            tier=request.tier,
                            min_quality_score=0.7,  # Only high-quality questions
                            limit=num_needed,
                            topic_keywords=topic_keywords,
                        )

                        if db_questions:
                            real_questions.extend(db_questions[:num_needed])
                            logger.info(
                                f"Retrieved {len(real_questions)} questions from database using "
                                f"syllabus refs {request.get_syllabus_refs()[:3]}... "
                                f"and topic keywords {topic_keywords}"
                            )

                    except Exception as e:
                        logger.error(f"Failed to retrieve questions from DB: {e}")

                # 2. If we still need more questions, generate them
                remaining_needed = num_needed - len(real_questions)
                if remaining_needed > 0 and self.question_generation_service:
                    try:
                        from src.models.question_models import GenerationRequest

                        generation_request = GenerationRequest(
                            topic=request.topic.value,
                            subtopics=request.subtopics,
                            subject_content_refs=request.get_syllabus_refs(),
                            tier=request.tier,
                            num_questions=remaining_needed,
                            difficulty=request.difficulty or "medium",
                            grade_level=request.grade_level or 8,
                        )

                        session = await self.question_generation_service.generate_questions(
                            generation_request
                        )

                        if session.questions:
                            real_questions.extend(session.questions[:remaining_needed])
                            logger.info(f"Generated {len(session.questions)} additional questions")

                    except Exception as e:
                        logger.error(f"Failed to generate questions: {e}")

                # 3. Replace LLM-generated questions with real ones
                if real_questions:
                    block.content["questions"] = [
                        self._extract_question_data(q) for q in real_questions
                    ]

                    logger.info(
                        f"Using {len(real_questions)} real questions ({len([q for q in real_questions if hasattr(q, 'question_id_global')])} from DB, {len(real_questions) - len([q for q in real_questions if hasattr(q, 'question_id_global')])} generated)"
                    )
                else:
                    logger.info(
                        "No real questions available, keeping LLM-generated placeholder questions"
                    )

    def _extract_question_data(self, question) -> dict[str, Any]:
        """Extract structured data from a Question object for document blocks."""
        try:
            # Basic question data
            question_data = {
                "text": question.raw_text_content,
                "marks": question.marks or getattr(question, "total_marks", 3),
                "difficulty": self._get_difficulty_from_question(question),
            }

            # Extract answer from solution and marking scheme
            if (
                hasattr(question, "solution_and_marking_scheme")
                and question.solution_and_marking_scheme
                and question.solution_and_marking_scheme.final_answers_summary
            ):
                answer = question.solution_and_marking_scheme.final_answers_summary[0].answer_text
                question_data["answer"] = answer

            # Add hint from first marking criterion if available
            if (
                hasattr(question, "solution_and_marking_scheme")
                and question.solution_and_marking_scheme
                and question.solution_and_marking_scheme.mark_allocation_criteria
            ):
                first_criterion = question.solution_and_marking_scheme.mark_allocation_criteria[0]
                if hasattr(first_criterion, "criterion_text"):
                    question_data["hint"] = f"Focus on: {first_criterion.criterion_text}"

            # Add topic information from taxonomy
            if hasattr(question, "taxonomy") and question.taxonomy:
                if hasattr(question.taxonomy, "skill_tags") and question.taxonomy.skill_tags:
                    question_data["topics"] = question.taxonomy.skill_tags
                if (
                    hasattr(question.taxonomy, "subject_content_references")
                    and question.taxonomy.subject_content_references
                ):
                    question_data["syllabus_refs"] = question.taxonomy.subject_content_references

            return question_data

        except Exception as e:
            logger.warning(f"Error extracting question data: {e}")
            # Fallback to basic data
            return {
                "text": getattr(question, "raw_text_content", "Question text not available"),
                "marks": getattr(question, "marks", 3),
                "difficulty": "medium",
            }

    def _get_difficulty_from_question(self, question) -> str:  # noqa: PLR0911
        """Extract difficulty from question object."""
        # Try direct difficulty level first
        if hasattr(question, "difficulty_level") and question.difficulty_level:
            return question.difficulty_level

        # Try numeric difficulty estimate
        if (
            hasattr(question, "taxonomy")
            and question.taxonomy
            and hasattr(question.taxonomy, "difficulty_estimate_0_to_1")
            and question.taxonomy.difficulty_estimate_0_to_1
        ):
            difficulty_score = question.taxonomy.difficulty_estimate_0_to_1
            if difficulty_score < 0.3:
                return "easy"
            if difficulty_score < 0.7:
                return "medium"
            return "hard"

        # Try cognitive level mapping
        if (
            hasattr(question, "taxonomy")
            and question.taxonomy
            and hasattr(question.taxonomy, "cognitive_level")
            and question.taxonomy.cognitive_level
        ):
            cognitive_level = question.taxonomy.cognitive_level.lower()
            if cognitive_level in ["recall", "procedural"]:
                return "easy"
            if cognitive_level in ["application", "conceptual"]:
                return "medium"
            return "hard"

        return "medium"  # Default

    def _calculate_word_count(self, rendered_blocks: dict[str, str]) -> int:
        """Calculate total word count from rendered content."""
        total_words = 0
        for content in rendered_blocks.values():
            # Simple word count (split by whitespace)
            total_words += len(content.split())
        return total_words

    def _map_detail_level_to_description(self, detail_level: int) -> str:
        """Map numeric detail level to descriptive string."""
        if detail_level <= 3:
            return "minimal"
        elif detail_level <= 6:
            return "medium"
        else:
            return "comprehensive"

    def _generate_document_tags(self, request: DocumentGenerationRequestV2) -> list[str]:
        """Generate tags from document request for categorization."""
        tags = []

        # Add document type
        tags.append(request.document_type.value)

        # Add topic (convert to kebab-case)
        topic_tag = request.topic.value.lower().replace(" ", "-").replace("&", "and")
        tags.append(topic_tag)

        # Add tier
        tags.append(request.tier.value.lower())

        # Add subtopics if provided
        if request.subtopics:
            tags.extend([subtopic.lower().replace(" ", "-") for subtopic in request.subtopics])

        return tags

    async def _create_document_metadata(
        self,
        request: DocumentGenerationRequestV2,
        document: GeneratedDocumentV2,
        session_id: str = None,
        status: str = "generated",
    ):
        """Create document metadata for storage."""
        from src.models.stored_document_models import StoredDocumentMetadata

        detail_description = self._map_detail_level_to_description(
            request.get_effective_detail_level()
        )
        tags = self._generate_document_tags(request)

        # Ensure estimated_duration is at least 1 (database constraint)
        estimated_duration = document.total_estimated_minutes or request.target_duration_minutes or 30
        estimated_duration = max(1, estimated_duration)  # Ensure it's at least 1

        metadata = StoredDocumentMetadata(
            session_id=session_id,
            title=document.title,
            document_type=request.document_type.value,
            detail_level=detail_description,
            topic=request.topic.value,
            grade_level=request.grade_level,
            estimated_duration=estimated_duration,
            total_questions=self._count_questions_in_document(document),
            status=status,
            tags=tags,
        )

        return metadata

    def _count_questions_in_document(self, document: GeneratedDocumentV2) -> int:
        """Count total questions in the document."""
        total_questions = 0

        for block_result in document.content_structure.blocks:
            if block_result.block_type == "practice_questions":
                questions = block_result.content.get("questions", [])
                total_questions += len(questions)

        return total_questions

    async def _save_document_to_storage(
        self,
        request: DocumentGenerationRequestV2,
        document: GeneratedDocumentV2,
        session_id: str = None,
    ) -> str:
        """Save generated document to storage repository."""
        if not self.document_storage_repository:
            logger.debug("No document storage repository configured, skipping save")
            return None

        try:
            # Create metadata
            metadata = await self._create_document_metadata(
                request, document, session_id, status="generated"
            )

            # Save to repository
            document_id = await self.document_storage_repository.save_document_metadata(metadata)

            logger.info(f"Successfully saved document metadata with ID: {document_id}")
            return str(document_id)

        except Exception as e:
            logger.error(f"Failed to save document to storage: {e}")
            # Don't fail the whole generation, just log the error
            return None

    async def _save_failed_document_to_storage(
        self, request: DocumentGenerationRequestV2, error_message: str
    ) -> str:
        """Save failed document metadata to storage."""
        if not self.document_storage_repository:
            return None

        try:
            from src.models.stored_document_models import StoredDocumentMetadata

            tags = self._generate_document_tags(request)

            metadata = StoredDocumentMetadata(
                title=request.title,
                document_type=request.document_type.value,
                detail_level=self._map_detail_level_to_description(
                    request.get_effective_detail_level()
                ),
                topic=request.topic.value,
                grade_level=request.grade_level,
                status="failed",
                tags=tags,
            )

            document_id = await self.document_storage_repository.save_document_metadata(metadata)
            logger.info(f"Saved failed document metadata with ID: {document_id}")
            return str(document_id)

        except Exception as e:
            logger.error(f"Failed to save failed document metadata: {e}")
            return None
