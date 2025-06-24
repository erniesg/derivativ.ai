"""
Diagram Storage Service - Abstraction layer for storing generated diagrams.

Provides a clean interface that can be swapped between local storage,
Supabase Storage, S3, etc. with minimal configuration changes.
"""

import json
import logging
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from src.models.diagram_models import DiagramGenerationResult

logger = logging.getLogger(__name__)


class DiagramStorageResult(BaseModel):
    """Result from storing a diagram"""

    success: bool = Field(..., description="Whether storage operation succeeded")
    question_id: str = Field(..., description="Question ID this diagram belongs to")
    diagram_path: Optional[str] = Field(None, description="Path to stored diagram image")
    manim_code_path: Optional[str] = Field(None, description="Path to stored Manim code")
    storage_type: str = Field(..., description="Type of storage used (local, supabase, s3)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Storage timestamp")

    def model_post_init(self, __context) -> None:
        """Validate paths are provided when success=True"""
        if self.success and (self.diagram_path is None or self.manim_code_path is None):
            raise ValueError("diagram_path and manim_code_path required when success=True")


class DiagramStorageInterface(ABC):
    """Abstract interface for diagram storage implementations"""

    @abstractmethod
    async def store_diagram(
        self, question_id: str, diagram_result: DiagramGenerationResult
    ) -> DiagramStorageResult:
        """
        Store a generated diagram and return storage information.

        Args:
            question_id: Unique identifier for the question
            diagram_result: Complete diagram generation result with Manim code

        Returns:
            DiagramStorageResult with storage details
        """
        pass

    @abstractmethod
    async def get_diagram_path(self, question_id: str) -> Optional[str]:
        """
        Get the path/URL to a stored diagram.

        Args:
            question_id: Question identifier

        Returns:
            Path or URL to diagram if exists, None otherwise
        """
        pass

    @abstractmethod
    async def diagram_exists(self, question_id: str) -> bool:
        """
        Check if a diagram exists for the given question.

        Args:
            question_id: Question identifier

        Returns:
            True if diagram exists, False otherwise
        """
        pass

    @abstractmethod
    async def get_diagram_metadata(self, question_id: str) -> Optional[dict[str, Any]]:
        """
        Get metadata for a stored diagram.

        Args:
            question_id: Question identifier

        Returns:
            Metadata dict if diagram exists, None otherwise
        """
        pass


class LocalDiagramStorageService(DiagramStorageInterface):
    """
    Local file system implementation of diagram storage.

    Stores diagrams as PNG files and maintains metadata in JSON format.
    """

    def __init__(
        self, base_path: Union[str, Path] = "generated_diagrams/", create_directories: bool = True
    ):
        """
        Initialize local storage service.

        Args:
            base_path: Base directory for storing diagrams
            create_directories: Whether to create directories if they don't exist
        """
        self.base_path = Path(base_path)
        self.storage_type = "local"

        if create_directories:
            self.base_path.mkdir(exist_ok=True, parents=True)

        # Metadata file for tracking all diagrams
        self.metadata_file = self.base_path / "metadata.json"
        self._ensure_metadata_file()

        logger.info(f"Initialized local diagram storage at: {self.base_path}")

    def _ensure_metadata_file(self):
        """Ensure metadata file exists with proper structure"""
        if not self.metadata_file.exists():
            initial_metadata = {
                "created_at": datetime.utcnow().isoformat(),
                "storage_type": "local",
                "diagrams": {},
            }
            with open(self.metadata_file, "w") as f:
                json.dump(initial_metadata, f, indent=2)

    def _validate_diagram_id(self, question_id: str) -> bool:
        """Validate question ID for safe file naming"""
        if not question_id or len(question_id) == 0:
            return False

        # Allow alphanumeric, hyphens, underscores only
        pattern = r"^[a-zA-Z0-9_-]+$"
        return bool(re.match(pattern, question_id))

    async def store_diagram(
        self, question_id: str, diagram_result: DiagramGenerationResult
    ) -> DiagramStorageResult:
        """Store diagram locally with Manim rendering"""

        if not self._validate_diagram_id(question_id):
            return DiagramStorageResult(
                success=False,
                question_id=question_id,
                storage_type=self.storage_type,
                error_message=f"Invalid question ID: {question_id}",
            )

        try:
            # Define file paths
            diagram_path = self.base_path / f"{question_id}.png"
            manim_code_path = self.base_path / f"{question_id}.py"

            # Write Manim code to file
            with open(manim_code_path, "w") as f:
                f.write(diagram_result.manim_code.manim_code)

            # Render Manim code to image
            render_success = await self._render_manim_to_image(
                manim_code_path, diagram_path, diagram_result.manim_code.scene_class_name
            )

            if not render_success:
                return DiagramStorageResult(
                    success=False,
                    question_id=question_id,
                    storage_type=self.storage_type,
                    error_message="Failed to render Manim code to image",
                )

            # Update metadata
            await self._update_metadata(question_id, diagram_result, diagram_path, manim_code_path)

            # Return success result
            return DiagramStorageResult(
                success=True,
                question_id=question_id,
                diagram_path=str(diagram_path),
                manim_code_path=str(manim_code_path),
                storage_type=self.storage_type,
                metadata={
                    "scene_class": diagram_result.manim_code.scene_class_name,
                    "complexity_score": diagram_result.manim_code.complexity_score,
                    "quality_score": diagram_result.validation_result.overall_quality,
                    "render_time": diagram_result.manim_code.estimated_render_time,
                },
            )

        except Exception as e:
            logger.error(f"Failed to store diagram for {question_id}: {e}")
            return DiagramStorageResult(
                success=False,
                question_id=question_id,
                storage_type=self.storage_type,
                error_message=f"Storage failed: {e!s}",
            )

    async def _render_manim_to_image(
        self, manim_file_path: Path, output_path: Path, scene_class_name: str
    ) -> bool:
        """
        Render Manim code to PNG image.

        Args:
            manim_file_path: Path to Manim .py file
            output_path: Path where PNG should be saved
            scene_class_name: Name of the Scene class to render

        Returns:
            True if rendering succeeded, False otherwise
        """
        try:
            # Create temporary directory for Manim output
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)

                # Manim command for generating PNG
                # --format=png creates static image
                # --media_dir specifies output directory
                # -ql for low quality (faster rendering)
                cmd = [
                    "manim",
                    str(manim_file_path),
                    scene_class_name,
                    "--format=png",
                    f"--media_dir={temp_dir_path}",
                    "-ql",  # Low quality for faster development
                ]

                # Run Manim command
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,  # 1 minute timeout
                    check=False,  # We handle return codes manually
                )

                if process.returncode != 0:
                    logger.error(f"Manim rendering failed: {process.stderr}")
                    return False

                # Find generated PNG file
                # Manim typically outputs to: temp_dir/images/scene_name/PNG/*.png
                png_files = list(temp_dir_path.glob("**/images/**/*.png"))

                if not png_files:
                    logger.error("No PNG file generated by Manim")
                    return False

                # Copy the generated PNG to our target location
                generated_png = png_files[0]  # Take first PNG found
                import shutil

                shutil.copy2(generated_png, output_path)

                logger.info(f"Successfully rendered diagram: {output_path}")
                return True

        except subprocess.TimeoutExpired:
            logger.error("Manim rendering timed out")
            return False
        except Exception as e:
            logger.error(f"Manim rendering error: {e}")
            return False

    async def _update_metadata(
        self,
        question_id: str,
        diagram_result: DiagramGenerationResult,
        diagram_path: Path,
        manim_code_path: Path,
    ):
        """Update metadata file with new diagram information"""
        try:
            # Load existing metadata
            with open(self.metadata_file) as f:
                metadata = json.load(f)

            # Add diagram entry
            metadata["diagrams"][question_id] = {
                "question_id": question_id,
                "diagram_path": str(diagram_path),
                "manim_code_path": str(manim_code_path),
                "scene_class_name": diagram_result.manim_code.scene_class_name,
                "storage_type": self.storage_type,
                "created_at": datetime.utcnow().isoformat(),
                "validation_result": diagram_result.validation_result.model_dump(),
                "cgv_metadata": {
                    "constraints": diagram_result.manifest_constraints.model_dump()
                    if diagram_result.manifest_constraints
                    else None,
                    "solution": diagram_result.geometric_solution.model_dump()
                    if diagram_result.geometric_solution
                    else None,
                    "generation_id": diagram_result.generation_id,
                    "processing_time": diagram_result.processing_time,
                },
            }

            # Update metadata
            metadata["updated_at"] = datetime.utcnow().isoformat()

            # Write back to file
            with open(self.metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to update metadata for {question_id}: {e}")

    async def get_diagram_path(self, question_id: str) -> Optional[str]:
        """Get path to stored diagram"""
        if not self._validate_diagram_id(question_id):
            return None

        diagram_path = self.base_path / f"{question_id}.png"
        return str(diagram_path) if diagram_path.exists() else None

    async def diagram_exists(self, question_id: str) -> bool:
        """Check if diagram exists"""
        if not self._validate_diagram_id(question_id):
            return False

        diagram_path = self.base_path / f"{question_id}.png"
        return diagram_path.exists()

    async def get_diagram_metadata(self, question_id: str) -> Optional[dict[str, Any]]:
        """Get metadata for stored diagram"""
        try:
            with open(self.metadata_file) as f:
                metadata = json.load(f)

            return metadata.get("diagrams", {}).get(question_id)

        except Exception as e:
            logger.warning(f"Failed to get metadata for {question_id}: {e}")
            return None


def create_storage_service(storage_type: str, **kwargs) -> DiagramStorageInterface:
    """
    Factory function to create storage service instances.

    Args:
        storage_type: Type of storage ("local", "supabase", "s3")
        **kwargs: Additional arguments for storage service

    Returns:
        DiagramStorageInterface implementation

    Raises:
        ValueError: If storage_type is not supported
    """
    if storage_type == "local":
        return LocalDiagramStorageService(**kwargs)
    elif storage_type == "supabase":
        # TODO: Implement SupabaseDiagramStorageService
        raise NotImplementedError("Supabase storage not yet implemented")
    elif storage_type == "s3":
        # TODO: Implement S3DiagramStorageService
        raise NotImplementedError("S3 storage not yet implemented")
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")


def create_storage_service_from_config() -> DiagramStorageInterface:
    """
    Create storage service from application configuration.

    Returns:
        Configured DiagramStorageInterface implementation
    """
    from src.core.config import get_settings

    settings = get_settings()

    # Default to local storage if not configured
    storage_type = getattr(settings, "diagram_storage_type", "local")
    base_path = getattr(settings, "diagram_base_path", "generated_diagrams/")

    return create_storage_service(storage_type, base_path=base_path, create_directories=True)
