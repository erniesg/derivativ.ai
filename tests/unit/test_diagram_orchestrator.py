"""
Unit tests for diagram orchestrator.
Tests the complete CGV pipeline orchestration with quality control.
"""

import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from src.models.diagram_models import (
    DiagramGenerationRequest,
    DiagramGenerationResult,
    DiagramValidationResult,
    GeometricSolution,
    ManifestConstraints,
    ManifestDiagramCode,
)
from src.models.enums import CommandWord
from src.models.question_models import (
    Question,
    QuestionTaxonomy,
    SolutionAndMarkingScheme,
    SolverAlgorithm,
)
from src.services.diagram_orchestrator import DiagramOrchestrator
from src.services.diagram_storage_service import DiagramStorageResult, LocalDiagramStorageService


class TestDiagramOrchestrator:
    """Test diagram orchestration with CGV pipeline"""

    @pytest.fixture
    def mock_constraint_service(self):
        """Mock constraint extraction service"""
        service = AsyncMock()
        service.extract_constraints_from_text.return_value = ManifestConstraints(
            question_id="test_q1", constraints=[], shapes=[], unknowns=[]
        )
        return service

    @pytest.fixture
    def mock_solver_service(self):
        """Mock geometric solver service"""
        service = AsyncMock()
        service.solve_constraints.return_value = GeometricSolution(
            is_valid=True, solved_points={}, solved_values={}
        )
        return service

    @pytest.fixture
    def mock_manim_service(self):
        """Mock Manim code generator service"""
        service = AsyncMock()
        service.generate_manim_code.return_value = ManifestDiagramCode(
            manim_code="# Sample Manim code",
            scene_class_name="TestScene",
            estimated_render_time=10.0,
            complexity_score=0.5,
        )
        return service

    @pytest.fixture
    def mock_validation_service(self):
        """Mock diagram validation service"""
        service = AsyncMock()
        service.validate_diagram.return_value = DiagramValidationResult(
            geometric_accuracy=0.9,
            readability_score=0.85,
            cambridge_compliance=0.95,
            label_placement_score=0.8,
            collision_detection_score=0.9,
            overall_quality=0.88,
        )
        return service

    @pytest.fixture
    def mock_storage_service(self):
        """Mock storage service"""
        service = AsyncMock()
        service.diagram_exists.return_value = False
        service.store_diagram.return_value = DiagramStorageResult(
            success=True,
            question_id="test_q1",
            diagram_path="/tmp/test_q1.png",
            manim_code_path="/tmp/test_q1.py",
            storage_type="local",
        )
        return service

    @pytest.fixture
    def orchestrator(
        self,
        mock_constraint_service,
        mock_solver_service,
        mock_manim_service,
        mock_validation_service,
        mock_storage_service,
    ):
        """Create orchestrator with mocked services"""
        return DiagramOrchestrator(
            constraint_service=mock_constraint_service,
            solver_service=mock_solver_service,
            manim_service=mock_manim_service,
            validation_service=mock_validation_service,
            storage_service=mock_storage_service,
        )

    @pytest.fixture
    def sample_geometry_question(self):
        """Create sample question that should trigger diagram generation"""
        return Question(
            question_id_local="1a",
            question_id_global="test_q1_geometry",
            question_number_display="1(a)",
            marks=5,
            command_word=CommandWord.CALCULATE,
            raw_text_content="Triangle ABC has AB = 10 cm, AC = 8 cm and angle BAC = 60°. Calculate the area of the triangle.",
            assets=[],
            taxonomy=QuestionTaxonomy(
                topic_path=["Geometry", "Triangles"],
                subject_content_references=[],
                skill_tags=["area_calculation"],
            ),
            solution_and_marking_scheme=SolutionAndMarkingScheme(
                final_answers_summary=[], mark_allocation_criteria=[], total_marks_for_part=5
            ),
            solver_algorithm=SolverAlgorithm(steps=[]),
        )

    @pytest.fixture
    def sample_non_geometry_question(self):
        """Create sample question that should NOT trigger diagram generation"""
        return Question(
            question_id_local="2a",
            question_id_global="test_q2_algebra",
            question_number_display="2(a)",
            marks=3,
            command_word=CommandWord.SOLVE,
            raw_text_content="Solve the equation 2x + 5 = 13 for x.",
            assets=[],
            taxonomy=QuestionTaxonomy(
                topic_path=["Algebra", "Linear Equations"],
                subject_content_references=[],
                skill_tags=["equation_solving"],
            ),
            solution_and_marking_scheme=SolutionAndMarkingScheme(
                final_answers_summary=[], mark_allocation_criteria=[], total_marks_for_part=3
            ),
            solver_algorithm=SolverAlgorithm(steps=[]),
        )

    def test_should_generate_diagram_geometry_question(
        self, orchestrator, sample_geometry_question
    ):
        """Test auto-detection of geometry questions"""
        should_generate, reason = orchestrator.should_generate_diagram(sample_geometry_question)

        assert should_generate is True
        # Should detect either keywords or geometric patterns
        assert any(
            keyword in reason.lower()
            for keyword in ["triangle", "geometric", "pattern", "keywords"]
        )

    def test_should_generate_diagram_non_geometry_question(
        self, orchestrator, sample_non_geometry_question
    ):
        """Test auto-detection rejects non-geometry questions"""
        should_generate, reason = orchestrator.should_generate_diagram(sample_non_geometry_question)

        assert should_generate is False
        assert "no geometric indicators" in reason.lower()

    def test_should_generate_diagram_with_exclusion_keywords(self, orchestrator):
        """Test exclusion keywords override geometry detection"""
        question_text = "Triangle ABC has sides 3, 4, 5. Note: no diagram shown."

        # Create question with exclusion keyword
        question = Question(
            question_id_local="3a",
            question_id_global="test_q3",
            question_number_display="3(a)",
            marks=4,
            command_word=CommandWord.FIND,
            raw_text_content=question_text,
            assets=[],
            taxonomy=QuestionTaxonomy(topic_path=[], subject_content_references=[], skill_tags=[]),
            solution_and_marking_scheme=SolutionAndMarkingScheme(
                final_answers_summary=[], mark_allocation_criteria=[], total_marks_for_part=4
            ),
            solver_algorithm=SolverAlgorithm(steps=[]),
        )

        should_generate, reason = orchestrator.should_generate_diagram(question)

        # Should be excluded despite having "triangle"
        assert should_generate is False
        assert "excluded" in reason.lower()

    async def test_generate_diagram_successful_high_quality(
        self, orchestrator, sample_geometry_question, mock_validation_service
    ):
        """Test successful diagram generation with high quality"""
        # Set high quality validation result
        mock_validation_service.validate_diagram.return_value = DiagramValidationResult(
            geometric_accuracy=0.95,
            readability_score=0.9,
            cambridge_compliance=0.95,
            label_placement_score=0.9,
            collision_detection_score=0.95,
            overall_quality=0.93,  # Above auto_approve_threshold (0.9)
        )

        success, asset, reason = await orchestrator.generate_diagram_for_question(
            sample_geometry_question
        )

        assert success is True
        assert asset is not None
        assert asset.asset_type == "diagram"
        assert "high quality" in reason.lower()

    async def test_generate_diagram_successful_acceptable_quality(
        self, orchestrator, sample_geometry_question, mock_validation_service
    ):
        """Test successful diagram generation with acceptable quality"""
        # Set acceptable quality validation result
        mock_validation_service.validate_diagram.return_value = DiagramValidationResult(
            geometric_accuracy=0.85,
            readability_score=0.8,
            cambridge_compliance=0.85,
            label_placement_score=0.8,
            collision_detection_score=0.8,
            overall_quality=0.82,  # Above min_threshold (0.8) but below auto_approve (0.9)
        )

        success, asset, reason = await orchestrator.generate_diagram_for_question(
            sample_geometry_question
        )

        assert success is True
        assert asset is not None
        assert "acceptable quality" in reason.lower()

    async def test_generate_diagram_quality_too_low(
        self, orchestrator, sample_geometry_question, mock_validation_service
    ):
        """Test diagram generation failure due to low quality"""
        # Set low quality validation result
        mock_validation_service.validate_diagram.return_value = DiagramValidationResult(
            geometric_accuracy=0.6,
            readability_score=0.5,
            cambridge_compliance=0.6,
            label_placement_score=0.5,
            collision_detection_score=0.6,
            overall_quality=0.55,  # Below min_threshold (0.8)
        )

        success, asset, reason = await orchestrator.generate_diagram_for_question(
            sample_geometry_question
        )

        assert success is False
        assert asset is None
        assert "quality too low" in reason.lower()

    async def test_generate_diagram_already_exists(
        self, orchestrator, sample_geometry_question, mock_storage_service
    ):
        """Test handling when diagram already exists"""
        # Mock existing diagram
        mock_storage_service.diagram_exists.return_value = True
        mock_storage_service.get_diagram_path.return_value = "/existing/path.png"
        mock_storage_service.get_diagram_metadata.return_value = {
            "question_id": "test_q1_geometry",
            "scene_class_name": "ExistingScene",
            "storage_type": "local",
        }

        success, asset, reason = await orchestrator.generate_diagram_for_question(
            sample_geometry_question
        )

        assert success is True
        assert asset is not None
        assert "already exists" in reason.lower()

    async def test_generate_diagram_auto_detection_disabled(
        self, orchestrator, sample_non_geometry_question
    ):
        """Test forcing diagram generation when auto-detection is disabled"""
        success, asset, reason = await orchestrator.generate_diagram_for_question(
            sample_non_geometry_question, force_generation=True
        )

        # Should attempt generation even for non-geometry question
        assert success is True  # Will succeed with mocked services

    async def test_generate_diagram_cgv_pipeline_failure(
        self, orchestrator, sample_geometry_question, mock_constraint_service
    ):
        """Test handling of CGV pipeline failures"""
        # Mock constraint service failure
        mock_constraint_service.extract_constraints_from_text.side_effect = Exception(
            "Constraint extraction failed"
        )

        success, asset, reason = await orchestrator.generate_diagram_for_question(
            sample_geometry_question
        )

        assert success is False
        assert asset is None
        assert "failed" in reason.lower()

    async def test_generate_diagram_storage_failure(
        self, orchestrator, sample_geometry_question, mock_storage_service
    ):
        """Test handling of storage failures"""
        # Mock storage failure
        mock_storage_service.store_diagram.return_value = DiagramStorageResult(
            success=False,
            question_id="test_q1_geometry",
            storage_type="local",
            error_message="Storage failed",
        )

        success, asset, reason = await orchestrator.generate_diagram_for_question(
            sample_geometry_question
        )

        assert success is False
        assert asset is None

    async def test_cgv_pipeline_execution(
        self,
        orchestrator,
        mock_constraint_service,
        mock_solver_service,
        mock_manim_service,
        mock_validation_service,
    ):
        """Test complete CGV pipeline execution"""
        request = DiagramGenerationRequest(
            question_text="Triangle ABC with AB = 5", question_id="test_cgv"
        )

        result = await orchestrator._execute_cgv_pipeline(request, 0)

        # Verify all services were called
        mock_constraint_service.extract_constraints_from_text.assert_called_once()
        mock_solver_service.solve_constraints.assert_called_once()
        mock_manim_service.generate_manim_code.assert_called_once()
        mock_validation_service.validate_diagram.assert_called_once()

        assert result.success is True
        assert result.validation_result.overall_quality == 0.88

    async def test_cgv_pipeline_execution_with_solver_failure(
        self, orchestrator, mock_solver_service
    ):
        """Test CGV pipeline handling of solver failures"""
        # Mock solver failure
        mock_solver_service.solve_constraints.return_value = GeometricSolution(
            is_valid=False, error_message="Geometric constraints unsolvable"
        )

        request = DiagramGenerationRequest(
            question_text="Invalid geometric constraints", question_id="test_cgv_fail"
        )

        result = await orchestrator._execute_cgv_pipeline(request, 0)

        assert result.success is False
        assert "unsolvable" in result.error_message.lower()

    def test_create_asset_from_result(self, orchestrator):
        """Test asset creation from diagram generation result"""
        # Create mock result
        validation_result = DiagramValidationResult(
            geometric_accuracy=0.9,
            readability_score=0.85,
            cambridge_compliance=0.95,
            label_placement_score=0.8,
            collision_detection_score=0.9,
            overall_quality=0.88,
        )

        result = DiagramGenerationResult(
            request=DiagramGenerationRequest(question_text="Test", question_id="test_asset"),
            manifest_constraints=ManifestConstraints(question_id="test_asset"),
            geometric_solution=GeometricSolution(),
            manim_code=ManifestDiagramCode(
                manim_code="# test code",
                scene_class_name="TestScene",
                estimated_render_time=10.0,
                complexity_score=0.5,
            ),
            validation_result=validation_result,
            success=True,
            quality_passed=True,
        )

        storage_result = DiagramStorageResult(
            success=True,
            question_id="test_asset",
            diagram_path="/test.png",
            manim_code_path="/test.py",
            storage_type="local",
        )

        asset = orchestrator._create_asset_from_result(result, storage_result)

        assert asset.asset_type == "diagram"
        assert asset.asset_id_local == "diagram_test_asset"
        assert asset.recreation_data is not None
        assert asset.recreation_data.manim_scene_class == "TestScene"

    @patch("src.services.diagram_orchestrator.create_storage_service_from_config")
    def test_orchestrator_with_real_storage_service(self, mock_create_storage):
        """Test orchestrator initialization with real storage service"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock storage service creation
            storage_service = LocalDiagramStorageService(base_path=temp_dir)
            mock_create_storage.return_value = storage_service

            # Create orchestrator without storage service (should use factory)
            orchestrator = DiagramOrchestrator(
                constraint_service=AsyncMock(),
                solver_service=AsyncMock(),
                manim_service=AsyncMock(),
                validation_service=AsyncMock(),
                # storage_service=None - will use factory
            )

            assert isinstance(orchestrator.storage_service, LocalDiagramStorageService)
            mock_create_storage.assert_called_once()


class TestDiagramOrchestratorFactory:
    """Test orchestrator factory function"""

    def test_create_diagram_orchestrator(self):
        """Test factory function creates orchestrator correctly"""
        from src.services.diagram_orchestrator import create_diagram_orchestrator

        constraint_service = AsyncMock()
        solver_service = AsyncMock()
        manim_service = AsyncMock()
        validation_service = AsyncMock()

        orchestrator = create_diagram_orchestrator(
            constraint_service=constraint_service,
            solver_service=solver_service,
            manim_service=manim_service,
            validation_service=validation_service,
        )

        assert isinstance(orchestrator, DiagramOrchestrator)
        assert orchestrator.constraint_service == constraint_service
        assert orchestrator.solver_service == solver_service
        assert orchestrator.manim_service == manim_service
        assert orchestrator.validation_service == validation_service
