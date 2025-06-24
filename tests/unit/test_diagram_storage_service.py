"""
Unit tests for diagram storage services.
Tests both abstract interface and local implementation with TDD approach.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.models.diagram_models import DiagramGenerationResult, ManifestDiagramCode
from src.services.diagram_storage_service import (
    DiagramStorageInterface,
    DiagramStorageResult,
    LocalDiagramStorageService,
    create_storage_service,
)


class TestDiagramStorageInterface:
    """Test the abstract storage interface contract"""

    def test_interface_methods_are_abstract(self):
        """Test that interface methods are properly abstract"""
        with pytest.raises(TypeError):
            # Should not be able to instantiate abstract class
            DiagramStorageInterface()


class TestLocalDiagramStorageService:
    """Test local file-based diagram storage implementation"""

    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary directory for storage tests"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def storage_service(self, temp_storage_dir):
        """Create local storage service with temp directory"""
        return LocalDiagramStorageService(base_path=temp_storage_dir)

    @pytest.fixture
    def sample_manim_code(self):
        """Sample Manim code for testing"""
        return """
from manim import *
import numpy as np

class TriangleABC(Scene):
    def construct(self):
        # NOT TO SCALE annotation
        not_to_scale = Text("NOT TO SCALE", font_size=16).to_edge(DOWN + RIGHT)

        # Define triangle vertices
        point_A = np.array([0, 0, 0])
        point_B = np.array([4, 0, 0])
        point_C = np.array([2, 3, 0])

        # Create triangle
        triangle = Polygon(point_A, point_B, point_C, color=BLACK)

        # Create labels
        label_A = MathTex("A", font_size=20).next_to(point_A, DOWN + LEFT)
        label_B = MathTex("B", font_size=20).next_to(point_B, DOWN + RIGHT)
        label_C = MathTex("C", font_size=20).next_to(point_C, UP)

        # Add side measurements
        side_AB = MathTex("10 cm", font_size=18).next_to(triangle.get_edge_center(point_A, point_B), DOWN)
        side_AC = MathTex("8 cm", font_size=18).next_to(triangle.get_edge_center(point_A, point_C), LEFT)

        # Add all elements
        self.add(triangle, label_A, label_B, label_C, side_AB, side_AC, not_to_scale)
        self.camera.background_color = WHITE
"""

    @pytest.fixture
    def sample_diagram_result(self, sample_manim_code):
        """Sample diagram generation result"""
        from src.models.diagram_models import (
            DiagramValidationResult,
            GeometricSolution,
            ManifestConstraints,
        )

        manim_code = ManifestDiagramCode(
            manim_code=sample_manim_code,
            scene_class_name="TriangleABC",
            estimated_render_time=15.0,
            complexity_score=0.6,
        )

        validation_result = DiagramValidationResult(
            geometric_accuracy=0.9,
            readability_score=0.85,
            cambridge_compliance=0.95,
            label_placement_score=0.8,
            collision_detection_score=0.9,
            overall_quality=0.88,
        )

        from src.models.diagram_models import DiagramGenerationRequest

        test_request = DiagramGenerationRequest(
            question_text="Sample test question", question_id="test_q1"
        )

        return DiagramGenerationResult(
            request=test_request,
            manifest_constraints=ManifestConstraints(question_id="test_q1"),
            geometric_solution=GeometricSolution(),
            manim_code=manim_code,
            validation_result=validation_result,
            success=True,
            quality_passed=True,
        )

    async def test_store_diagram_creates_files(
        self, storage_service, temp_storage_dir, sample_diagram_result
    ):
        """Test that storing a diagram creates the expected files"""
        question_id = "test_q1"

        # Mock the Manim rendering process
        with patch.object(
            storage_service, "_render_manim_to_image", new_callable=AsyncMock
        ) as mock_render:
            mock_render.return_value = True

            # Store diagram
            result = await storage_service.store_diagram(question_id, sample_diagram_result)

            # Verify result
            assert result.success is True
            assert result.diagram_path is not None
            assert result.manim_code_path is not None
            assert result.question_id == question_id

            # Verify files exist
            assert Path(result.diagram_path).exists()
            assert Path(result.manim_code_path).exists()

            # Verify Manim code content
            with open(result.manim_code_path) as f:
                stored_code = f.read()
            assert "TriangleABC" in stored_code
            assert "NOT TO SCALE" in stored_code

    async def test_get_diagram_path_existing_file(self, storage_service, temp_storage_dir):
        """Test retrieving path for existing diagram"""
        question_id = "test_q2"

        # Create a dummy diagram file
        diagram_path = temp_storage_dir / f"{question_id}.png"
        diagram_path.write_text("dummy image data")

        # Test retrieval
        result_path = await storage_service.get_diagram_path(question_id)
        assert result_path == str(diagram_path)

    async def test_get_diagram_path_nonexistent_file(self, storage_service):
        """Test retrieving path for non-existent diagram"""
        question_id = "nonexistent_q"

        result_path = await storage_service.get_diagram_path(question_id)
        assert result_path is None

    async def test_diagram_exists_check(self, storage_service, temp_storage_dir):
        """Test checking if diagram exists"""
        question_id = "test_q3"

        # Initially should not exist
        exists = await storage_service.diagram_exists(question_id)
        assert exists is False

        # Create diagram file
        diagram_path = temp_storage_dir / f"{question_id}.png"
        diagram_path.write_text("dummy image data")

        # Now should exist
        exists = await storage_service.diagram_exists(question_id)
        assert exists is True

    async def test_get_diagram_metadata(
        self, storage_service, temp_storage_dir, sample_diagram_result
    ):
        """Test retrieving diagram metadata"""
        question_id = "test_q4"

        # Mock rendering and store diagram
        with patch.object(
            storage_service, "_render_manim_to_image", new_callable=AsyncMock
        ) as mock_render:
            mock_render.return_value = True

            # Store diagram
            await storage_service.store_diagram(question_id, sample_diagram_result)

            # Get metadata
            metadata = await storage_service.get_diagram_metadata(question_id)

            assert metadata is not None
            assert metadata["question_id"] == question_id
            assert metadata["storage_type"] == "local"
            assert "created_at" in metadata
            assert "validation_result" in metadata
            assert metadata["validation_result"]["overall_quality"] == 0.88

    async def test_store_diagram_render_failure(self, storage_service, sample_diagram_result):
        """Test handling of Manim render failures"""
        question_id = "test_q_fail"

        # Mock render failure
        with patch.object(
            storage_service, "_render_manim_to_image", new_callable=AsyncMock
        ) as mock_render:
            mock_render.return_value = False

            # Store diagram should handle failure gracefully
            result = await storage_service.store_diagram(question_id, sample_diagram_result)

            assert result.success is False
            assert result.error_message is not None
            assert "render" in result.error_message.lower()

    async def test_cleanup_temp_files(
        self, storage_service, temp_storage_dir, sample_diagram_result
    ):
        """Test that temporary files are cleaned up after rendering"""
        question_id = "test_q_cleanup"

        with patch.object(
            storage_service, "_render_manim_to_image", new_callable=AsyncMock
        ) as mock_render:
            mock_render.return_value = True

            # Track temp directory before and after
            initial_files = set(temp_storage_dir.iterdir())

            await storage_service.store_diagram(question_id, sample_diagram_result)

            final_files = set(temp_storage_dir.iterdir())

            # Should only have the final diagram and manim code files
            new_files = final_files - initial_files
            assert len(new_files) == 3  # PNG, .py, metadata.json

            # Check file types
            file_extensions = {f.suffix for f in new_files}
            assert ".png" in file_extensions
            assert ".py" in file_extensions
            assert ".json" in file_extensions

    async def test_concurrent_storage_operations(self, storage_service, sample_diagram_result):
        """Test handling of concurrent storage operations"""
        import asyncio

        question_ids = ["concurrent_q1", "concurrent_q2", "concurrent_q3"]

        with patch.object(
            storage_service, "_render_manim_to_image", new_callable=AsyncMock
        ) as mock_render:
            mock_render.return_value = True

            # Store multiple diagrams concurrently
            tasks = [
                storage_service.store_diagram(qid, sample_diagram_result) for qid in question_ids
            ]

            results = await asyncio.gather(*tasks)

            # All should succeed
            assert all(result.success for result in results)
            assert len(set(result.question_id for result in results)) == 3

    def test_validate_diagram_id(self, storage_service):
        """Test diagram ID validation"""
        # Valid IDs
        assert storage_service._validate_diagram_id("test_q1") is True
        assert storage_service._validate_diagram_id("q123_abc") is True
        assert storage_service._validate_diagram_id("valid-id_123") is True

        # Invalid IDs
        assert storage_service._validate_diagram_id("") is False
        assert storage_service._validate_diagram_id("../invalid") is False
        assert storage_service._validate_diagram_id("id with spaces") is False
        assert storage_service._validate_diagram_id("id/with/slashes") is False


class TestStorageServiceFactory:
    """Test the storage service factory function"""

    def test_create_local_storage_service(self):
        """Test creating local storage service via factory"""
        service = create_storage_service("local", base_path="/tmp/test")
        assert isinstance(service, LocalDiagramStorageService)
        assert str(service.base_path) == "/tmp/test"

    def test_create_storage_service_with_config(self):
        """Test creating storage service with config dict"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"type": "local", "base_path": temp_dir, "create_directories": True}

            service = create_storage_service(config["type"], base_path=config["base_path"])
            assert isinstance(service, LocalDiagramStorageService)

    def test_create_storage_service_invalid_type(self):
        """Test error handling for invalid storage type"""
        with pytest.raises(ValueError, match="Unsupported storage type"):
            create_storage_service("invalid_type")

    @patch("src.core.config.get_settings")
    def test_create_storage_service_from_config(self, mock_get_settings):
        """Test creating storage service from application config"""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock config settings
            mock_settings = AsyncMock()
            mock_settings.diagram_storage_type = "local"
            mock_settings.diagram_base_path = temp_dir
            mock_get_settings.return_value = mock_settings

            from src.services.diagram_storage_service import create_storage_service_from_config

            service = create_storage_service_from_config()
            assert isinstance(service, LocalDiagramStorageService)


class TestDiagramStorageError:
    """Test error handling and edge cases"""

    def test_storage_result_model_validation(self):
        """Test DiagramStorageResult model validation"""
        # Valid result
        result = DiagramStorageResult(
            success=True,
            question_id="test_q",
            diagram_path="/path/to/diagram.png",
            manim_code_path="/path/to/code.py",
            storage_type="local",
        )
        assert result.success is True
        assert result.error_message is None

        # Invalid result (missing required fields for success=True)
        with pytest.raises(ValueError):
            DiagramStorageResult(
                success=True,
                question_id="test_q",
                storage_type="local",
                # Missing diagram_path - should fail validation
            )
