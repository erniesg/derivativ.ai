"""
Diagram Generation Orchestrator - Coordinates the full CGV pipeline.

Implements the Constrain-Generate-Verify workflow with quality control loops,
automatic diagram detection, and intelligent retry logic.
"""

import logging
import re
from typing import Optional

from src.core.config import load_config
from src.models.diagram_models import (
    DiagramGenerationRequest,
    DiagramGenerationResult,
    DiagramValidationResult,
)
from src.models.question_models import AssetRecreationData, Question, QuestionAsset
from src.services.constraint_extraction_service import ConstraintExtractionService
from src.services.diagram_storage_service import (
    DiagramStorageInterface,
    create_storage_service_from_config,
)
from src.services.diagram_validation_service import DiagramValidationService
from src.services.geometric_solver_service import GeometricSolverService
from src.services.manim_code_generator_service import ManimCodeGeneratorService

logger = logging.getLogger(__name__)


class DiagramOrchestrator:
    """
    Orchestrates the complete diagram generation workflow.

    Implements the CGV (Constrain-Generate-Verify) pipeline with:
    - Automatic geometry detection in question text
    - Quality-controlled generation with retry logic
    - Configurable storage backend switching
    - Manual and automated operation modes
    """

    def __init__(
        self,
        constraint_service: ConstraintExtractionService,
        solver_service: GeometricSolverService,
        manim_service: ManimCodeGeneratorService,
        validation_service: DiagramValidationService,
        storage_service: Optional[DiagramStorageInterface] = None,
    ):
        """
        Initialize diagram orchestrator with CGV services.

        Args:
            constraint_service: Service for constraint extraction (CONSTRAIN)
            solver_service: Service for geometric solving (GENERATE)
            manim_service: Service for Manim code generation (GENERATE)
            validation_service: Service for diagram validation (VERIFY)
            storage_service: Storage service (auto-created from config if None)
        """
        self.constraint_service = constraint_service
        self.solver_service = solver_service
        self.manim_service = manim_service
        self.validation_service = validation_service
        self.storage_service = storage_service or create_storage_service_from_config()

        # Load configuration
        self.config = load_config()
        # Get diagram config from config file, since AppConfig doesn't have diagram_generation field yet
        from src.core.config import get_config_manager

        config_manager = get_config_manager()
        raw_config = config_manager._load_yaml_config()
        self.diagram_config = raw_config.get("diagram_generation", {})

        # Quality control settings
        quality_config = self.diagram_config.get("quality_control", {})
        self.min_quality_threshold = quality_config.get("min_quality_threshold", 0.8)
        self.max_retry_attempts = quality_config.get("max_retry_attempts", 3)
        self.auto_approve_threshold = quality_config.get("auto_approve_threshold", 0.9)

        # Auto-detection settings
        auto_config = self.diagram_config.get("auto_detection", {})
        self.auto_detection_enabled = auto_config.get("enabled", True)
        self.geometry_keywords = set(auto_config.get("geometry_keywords", []))
        self.exclude_keywords = set(auto_config.get("exclude_keywords", []))

        logger.info("Diagram orchestrator initialized with CGV pipeline")

    def should_generate_diagram(self, question: Question) -> tuple[bool, str]:
        """
        Determine if a diagram should be generated for this question.

        Args:
            question: Question to analyze

        Returns:
            Tuple of (should_generate, reasoning)
        """
        text = question.raw_text_content.lower()

        # Check for explicit exclusions first
        for exclude_word in self.exclude_keywords:
            if exclude_word in text:
                return False, f"Excluded by keyword: '{exclude_word}'"

        # Check for geometry indicators
        found_keywords = []
        for keyword in self.geometry_keywords:
            if keyword in text:
                found_keywords.append(keyword)

        if found_keywords:
            return True, f"Geometry keywords detected: {', '.join(found_keywords)}"

        # Additional heuristics
        # Look for geometric notation patterns
        geometric_patterns = [
            r"\b[A-Z]{2,4}\b",  # Point/line notation like AB, ABC, ABCD
            r"\d+\s*(?:cm|mm|m|degrees?|°)",  # Measurements
            r"angle\s+[A-Z]{3}",  # Angle notation
            r"triangle\s+[A-Z]{3}",  # Triangle notation
        ]

        for pattern in geometric_patterns:
            if re.search(pattern, text):
                return True, f"Geometric notation pattern detected: {pattern}"

        return False, "No geometric indicators found"

    async def generate_diagram_for_question(  # noqa: PLR0911
        self, question: Question, force_generation: bool = False
    ) -> tuple[bool, Optional[QuestionAsset], str]:
        """
        Generate diagram for a question using CGV pipeline with quality control.

        Args:
            question: Question to generate diagram for
            force_generation: Skip auto-detection and force diagram generation

        Returns:
            Tuple of (success, asset_if_successful, reasoning)
        """
        question_id = question.question_id_global

        # Check if diagram already exists
        if await self.storage_service.diagram_exists(question_id):
            existing_path = await self.storage_service.get_diagram_path(question_id)
            logger.info(f"Diagram already exists for {question_id}: {existing_path}")

            # Create asset from existing diagram
            metadata = await self.storage_service.get_diagram_metadata(question_id)
            asset = self._create_asset_from_stored_diagram(question_id, existing_path, metadata)
            return True, asset, "Diagram already exists"

        # Auto-detection check (unless forced)
        if not force_generation and self.auto_detection_enabled:
            should_generate, detection_reason = self.should_generate_diagram(question)
            if not should_generate:
                logger.info(f"Skipping diagram for {question_id}: {detection_reason}")
                return False, None, f"Auto-detection: {detection_reason}"

        # Create generation request
        request = DiagramGenerationRequest(
            question_text=question.raw_text_content,
            question_id=question_id,
            include_labels=True,
            include_measurements=True,
        )

        # Execute CGV pipeline with retry logic
        for attempt in range(self.max_retry_attempts):
            try:
                logger.info(f"CGV attempt {attempt + 1} for {question_id}")

                # Execute single CGV attempt
                result = await self._execute_cgv_pipeline(request, attempt)

                if not result.success:
                    logger.warning(f"CGV attempt {attempt + 1} failed: {result.error_message}")
                    continue

                # Check quality threshold
                quality_score = result.validation_result.overall_quality

                if quality_score >= self.auto_approve_threshold:
                    # High quality - auto approve and store
                    logger.info(f"High quality diagram generated (score: {quality_score:.3f})")

                    storage_result = await self.storage_service.store_diagram(question_id, result)
                    if storage_result.success:
                        asset = self._create_asset_from_result(result, storage_result)
                        return True, asset, f"High quality generation (score: {quality_score:.3f})"
                    else:
                        logger.error(f"Storage failed: {storage_result.error_message}")
                        continue

                elif quality_score >= self.min_quality_threshold:
                    # Acceptable quality - store and return
                    logger.info(
                        f"Acceptable quality diagram generated (score: {quality_score:.3f})"
                    )

                    storage_result = await self.storage_service.store_diagram(question_id, result)
                    if storage_result.success:
                        asset = self._create_asset_from_result(result, storage_result)
                        return (
                            True,
                            asset,
                            f"Acceptable quality generation (score: {quality_score:.3f})",
                        )
                    else:
                        logger.error(f"Storage failed: {storage_result.error_message}")
                        continue

                else:
                    # Below threshold - retry if attempts remaining
                    logger.warning(
                        f"Quality too low (score: {quality_score:.3f} < {self.min_quality_threshold:.3f})"
                    )
                    if attempt < self.max_retry_attempts - 1:
                        logger.info(f"Retrying with adjusted parameters (attempt {attempt + 2})")
                        continue
                    else:
                        return (
                            False,
                            None,
                            f"Quality too low after {self.max_retry_attempts} attempts",
                        )

            except Exception as e:
                logger.error(f"CGV attempt {attempt + 1} failed with exception: {e}")
                if attempt == self.max_retry_attempts - 1:
                    return False, None, f"All attempts failed. Last error: {e!s}"
                continue

        return False, None, "All CGV attempts failed"

    async def _execute_cgv_pipeline(
        self, request: DiagramGenerationRequest, attempt: int
    ) -> DiagramGenerationResult:
        """
        Execute a single CGV pipeline attempt.

        Args:
            request: Diagram generation request
            attempt: Attempt number (0-indexed)

        Returns:
            DiagramGenerationResult with pipeline results
        """
        try:
            # CONSTRAIN: Extract constraints from question text
            logger.debug("CGV CONSTRAIN: Extracting constraints from question text")
            manifest_constraints = await self.constraint_service.extract_constraints_from_text(
                request.question_text, request.question_id or "unknown"
            )

            # GENERATE: Solve geometric constraints
            logger.debug("CGV GENERATE: Solving geometric constraints")
            geometric_solution = await self.solver_service.solve_constraints(manifest_constraints)

            if not geometric_solution.is_valid:
                raise ValueError(f"Geometric solving failed: {geometric_solution.error_message}")

            # GENERATE: Create Manim code
            logger.debug("CGV GENERATE: Creating Manim code")
            manim_code = await self.manim_service.generate_manim_code(
                manifest_constraints, geometric_solution
            )

            # VERIFY: Validate diagram quality
            logger.debug("CGV VERIFY: Validating diagram quality")
            validation_result = await self.validation_service.validate_diagram(
                manim_code, manifest_constraints, geometric_solution
            )

            # Create successful result
            result = DiagramGenerationResult(
                request=request,
                manifest_constraints=manifest_constraints,
                geometric_solution=geometric_solution,
                manim_code=manim_code,
                validation_result=validation_result,
                success=True,
                quality_passed=validation_result.overall_quality >= self.min_quality_threshold,
                processing_time=0.0,  # TODO: Add timing
                agent_reasoning_steps=[
                    {
                        "step": "CONSTRAIN",
                        "result": "constraints_extracted",
                        "count": len(manifest_constraints.constraints),
                    },
                    {
                        "step": "GENERATE",
                        "result": "geometry_solved",
                        "valid": geometric_solution.is_valid,
                    },
                    {
                        "step": "GENERATE",
                        "result": "manim_generated",
                        "scene": manim_code.scene_class_name,
                    },
                    {
                        "step": "VERIFY",
                        "result": "validation_complete",
                        "quality": validation_result.overall_quality,
                    },
                ],
            )

            logger.info(
                f"CGV pipeline completed successfully (quality: {validation_result.overall_quality:.3f})"
            )
            return result

        except Exception as e:
            # Create failed result
            logger.error(f"CGV pipeline failed: {e}")

            # Create default empty objects for failed result
            from src.models.diagram_models import (
                GeometricSolution,
                ManifestConstraints,
                ManifestDiagramCode,
            )

            default_constraints = ManifestConstraints(
                question_id=request.question_id or "unknown", constraints=[], shapes=[], unknowns=[]
            )

            default_solution = GeometricSolution(
                is_valid=False, error_message=str(e), solved_points={}, solved_values={}
            )

            default_manim_code = ManifestDiagramCode(
                manim_code="# Failed to generate diagram code",
                scene_class_name="FailedGeneration",
                estimated_render_time=0.0,
                complexity_score=0.0,
            )

            return DiagramGenerationResult(
                request=request,
                manifest_constraints=default_constraints,
                geometric_solution=default_solution,
                manim_code=default_manim_code,
                validation_result=DiagramValidationResult(
                    geometric_accuracy=0.0,
                    readability_score=0.0,
                    cambridge_compliance=0.0,
                    label_placement_score=0.0,
                    collision_detection_score=0.0,
                    overall_quality=0.0,
                    validation_issues=[f"Pipeline failed: {e!s}"],
                    improvement_suggestions=["Review input question for geometric content"],
                ),
                success=False,
                error_message=str(e),
                quality_passed=False,
            )

    def _create_asset_from_result(
        self, result: DiagramGenerationResult, storage_result
    ) -> QuestionAsset:
        """Create QuestionAsset from successful diagram generation"""

        asset_recreation_data = AssetRecreationData(
            diagram_type="geometric",
            manim_script_path=storage_result.manim_code_path,
            manim_scene_class=result.manim_code.scene_class_name,
            cgv_metadata={
                "storage_info": {
                    "storage_type": storage_result.storage_type,
                    "diagram_path": storage_result.diagram_path,
                    "created_at": storage_result.created_at.isoformat(),
                },
                "constraints_extracted": result.manifest_constraints.model_dump()
                if result.manifest_constraints
                else None,
                "geometric_solution": result.geometric_solution.model_dump()
                if result.geometric_solution
                else None,
                "validation_result": result.validation_result.model_dump(),
                "generation_attempts": 1,  # TODO: Track actual attempts
                "quality_score": result.validation_result.overall_quality,
            },
        )

        return QuestionAsset(
            asset_id_local=f"diagram_{result.request.question_id}",
            asset_type="diagram",
            description_for_accessibility=f"Geometric diagram for question {result.request.question_id}",
            recreation_data=asset_recreation_data,
        )

    def _create_asset_from_stored_diagram(
        self, question_id: str, diagram_path: str, metadata: Optional[dict]
    ) -> QuestionAsset:
        """Create QuestionAsset from existing stored diagram"""

        cgv_metadata = metadata.get("cgv_metadata", {}) if metadata else {}

        asset_recreation_data = AssetRecreationData(
            diagram_type="geometric",
            manim_script_path=metadata.get("manim_code_path") if metadata else None,
            manim_scene_class=metadata.get("scene_class_name") if metadata else None,
            cgv_metadata=cgv_metadata,
        )

        return QuestionAsset(
            asset_id_local=f"diagram_{question_id}",
            asset_type="diagram",
            description_for_accessibility=f"Geometric diagram for question {question_id}",
            recreation_data=asset_recreation_data,
        )


# Factory function for easy dependency injection
def create_diagram_orchestrator(
    constraint_service: ConstraintExtractionService,
    solver_service: GeometricSolverService,
    manim_service: ManimCodeGeneratorService,
    validation_service: DiagramValidationService,
) -> DiagramOrchestrator:
    """
    Factory function to create diagram orchestrator with all dependencies.

    Args:
        constraint_service: Constraint extraction service
        solver_service: Geometric solver service
        manim_service: Manim code generator service
        validation_service: Diagram validation service

    Returns:
        Configured DiagramOrchestrator instance
    """
    return DiagramOrchestrator(
        constraint_service=constraint_service,
        solver_service=solver_service,
        manim_service=manim_service,
        validation_service=validation_service,
    )
