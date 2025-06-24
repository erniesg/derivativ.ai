"""
Unit tests for diagram generation models.
Tests Pydantic model validation, constraint extraction schemas, and CGV pipeline data flow.
"""

from datetime import datetime

import pytest

from src.models.diagram_models import (
    CoordinatePoint,
    DiagramCanvas,
    DiagramGenerationRequest,
    DiagramGenerationResult,
    DiagramValidationResult,
    GeometricConstraint,
    GeometricConstraintType,
    GeometricSolution,
    ManifestConstraints,
    ManifestDiagramCode,
    ShapeDefinition,
    ShapeType,
)


class TestGeometricConstraint:
    """Test geometric constraint model validation"""

    def test_equality_constraint_creation(self):
        """Test creating an equality constraint (AB = 10)"""
        constraint = GeometricConstraint(
            constraint_id="c1",
            constraint_type=GeometricConstraintType.EQUALITY,
            target_elements=["side_AB"],
            value=10.0,
            unit="cm",
            text_source="AB = 10 cm",
            confidence=0.95,
        )

        assert constraint.constraint_type == GeometricConstraintType.EQUALITY
        assert constraint.target_elements == ["side_AB"]
        assert constraint.value == 10.0
        assert constraint.unit == "cm"
        assert constraint.confidence == 0.95

    def test_angle_constraint_creation(self):
        """Test creating an angle constraint (angle ABC = 60°)"""
        constraint = GeometricConstraint(
            constraint_id="c2",
            constraint_type=GeometricConstraintType.ANGLE,
            target_elements=["angle_ABC"],
            value=60.0,
            unit="degrees",
            text_source="angle ABC = 60°",
        )

        assert constraint.constraint_type == GeometricConstraintType.ANGLE
        assert constraint.target_elements == ["angle_ABC"]
        assert constraint.value == 60.0
        assert constraint.unit == "degrees"

    def test_constraint_validation_empty_elements(self):
        """Test that empty target_elements raises validation error"""
        with pytest.raises(ValueError, match="target_elements cannot be empty"):
            GeometricConstraint(
                constraint_id="c3",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=[],
                text_source="Invalid constraint",
            )

    def test_constraint_validation_confidence_range(self):
        """Test confidence value validation (0.0 to 1.0)"""
        # Valid confidence
        constraint = GeometricConstraint(
            constraint_id="c4",
            constraint_type=GeometricConstraintType.ANGLE,
            target_elements=["angle_ABC"],
            text_source="Test",
            confidence=0.5,
        )
        assert constraint.confidence == 0.5

        # Invalid confidence > 1.0
        with pytest.raises(ValueError):
            GeometricConstraint(
                constraint_id="c5",
                constraint_type=GeometricConstraintType.ANGLE,
                target_elements=["angle_ABC"],
                text_source="Test",
                confidence=1.5,
            )


class TestShapeDefinition:
    """Test shape definition model validation"""

    def test_triangle_creation(self):
        """Test creating a valid triangle"""
        triangle = ShapeDefinition(
            shape_id="triangle1",
            shape_type=ShapeType.TRIANGLE,
            vertices=["A", "B", "C"],
            properties={"type": "scalene"},
        )

        assert triangle.shape_type == ShapeType.TRIANGLE
        assert triangle.vertices == ["A", "B", "C"]
        assert triangle.properties["type"] == "scalene"

    def test_triangle_wrong_vertex_count(self):
        """Test that triangle with wrong vertex count fails validation"""
        with pytest.raises(ValueError, match="Triangle must have exactly 3 vertices"):
            ShapeDefinition(
                shape_id="invalid_triangle",
                shape_type=ShapeType.TRIANGLE,
                vertices=["A", "B"],  # Only 2 vertices
            )

    def test_quadrilateral_creation(self):
        """Test creating a valid quadrilateral"""
        quad = ShapeDefinition(
            shape_id="quad1",
            shape_type=ShapeType.QUADRILATERAL,
            vertices=["A", "B", "C", "D"],
            properties={"type": "rectangle"},
        )

        assert quad.shape_type == ShapeType.QUADRILATERAL
        assert len(quad.vertices) == 4

    def test_quadrilateral_wrong_vertex_count(self):
        """Test that quadrilateral with wrong vertex count fails validation"""
        with pytest.raises(ValueError, match="Quadrilateral must have exactly 4 vertices"):
            ShapeDefinition(
                shape_id="invalid_quad",
                shape_type=ShapeType.QUADRILATERAL,
                vertices=["A", "B", "C"],  # Only 3 vertices
            )


class TestDiagramCanvas:
    """Test diagram canvas configuration"""

    def test_default_canvas(self):
        """Test default canvas configuration"""
        canvas = DiagramCanvas()

        assert canvas.width == 16.0
        assert canvas.height == 9.0
        assert canvas.background_color == "WHITE"
        assert canvas.scale_factor == 1.0
        assert canvas.margin == 0.5

    def test_custom_canvas(self):
        """Test custom canvas configuration"""
        canvas = DiagramCanvas(
            width=12.0, height=8.0, background_color="LIGHT_GRAY", scale_factor=1.5, margin=1.0
        )

        assert canvas.width == 12.0
        assert canvas.height == 8.0
        assert canvas.scale_factor == 1.5

    def test_canvas_validation_positive_scale(self):
        """Test that scale factor must be positive"""
        with pytest.raises(ValueError):
            DiagramCanvas(scale_factor=0)

        with pytest.raises(ValueError):
            DiagramCanvas(scale_factor=-0.5)


class TestCoordinatePoint:
    """Test coordinate point model"""

    def test_point_creation(self):
        """Test creating a coordinate point"""
        point = CoordinatePoint(x=3.5, y=-2.1, label="A")

        assert point.x == 3.5
        assert point.y == -2.1
        assert point.label == "A"

    def test_point_to_numpy(self):
        """Test conversion to numpy array for Manim"""
        point = CoordinatePoint(x=1.0, y=2.0)
        np_array = point.to_numpy()

        assert np_array.tolist() == [1.0, 2.0, 0]
        assert np_array.shape == (3,)


class TestManifestConstraints:
    """Test manifest constraints (Constrain step output)"""

    def test_manifest_creation(self):
        """Test creating manifest constraints"""
        triangle = ShapeDefinition(
            shape_id="tri1", shape_type=ShapeType.TRIANGLE, vertices=["A", "B", "C"]
        )

        constraint = GeometricConstraint(
            constraint_id="c1",
            constraint_type=GeometricConstraintType.EQUALITY,
            target_elements=["side_AB"],
            value=10.0,
            text_source="AB = 10",
        )

        manifest = ManifestConstraints(
            question_id="q123", shapes=[triangle], constraints=[constraint], unknowns=["angle_ABC"]
        )

        assert manifest.question_id == "q123"
        assert len(manifest.shapes) == 1
        assert len(manifest.constraints) == 1
        assert manifest.unknowns == ["angle_ABC"]
        assert isinstance(manifest.extracted_at, datetime)


class TestGeometricSolution:
    """Test geometric solution (Generate step output)"""

    def test_solution_creation(self):
        """Test creating geometric solution"""
        point_a = CoordinatePoint(x=0, y=0, label="A")
        point_b = CoordinatePoint(x=3, y=4, label="B")

        solution = GeometricSolution(
            solved_points={"A": point_a, "B": point_b},
            solved_values={"angle_ABC": 60.0, "side_AB": 5.0},
            is_valid=True,
        )

        assert len(solution.solved_points) == 2
        assert solution.solved_values["angle_ABC"] == 60.0
        assert solution.is_valid is True
        assert solution.solution_method == "sympy_solving"

    def test_solution_with_error(self):
        """Test solution with error state"""
        solution = GeometricSolution(
            is_valid=False, error_message="Inconsistent constraints: triangle inequality violated"
        )

        assert solution.is_valid is False
        assert "triangle inequality" in solution.error_message


class TestManifestDiagramCode:
    """Test Manim code generation model"""

    def test_manim_code_creation(self):
        """Test creating Manim code manifest"""
        manim_code = ManifestDiagramCode(
            manim_code="class TestScene(Scene):\n    def construct(self):\n        pass",
            scene_class_name="TestScene",
            complexity_score=0.7,
            estimated_render_time=25.0,
        )

        assert "class TestScene" in manim_code.manim_code
        assert manim_code.scene_class_name == "TestScene"
        assert manim_code.complexity_score == 0.7
        assert manim_code.estimated_render_time == 25.0
        assert "manim" in manim_code.dependencies

    def test_complexity_score_validation(self):
        """Test complexity score range validation"""
        # Valid complexity score
        code = ManifestDiagramCode(manim_code="test", scene_class_name="Test", complexity_score=0.5)
        assert code.complexity_score == 0.5

        # Invalid complexity score > 1.0
        with pytest.raises(ValueError):
            ManifestDiagramCode(manim_code="test", scene_class_name="Test", complexity_score=1.5)


class TestDiagramValidationResult:
    """Test diagram validation (Verify step output)"""

    def test_validation_creation(self):
        """Test creating validation result"""
        validation = DiagramValidationResult(
            geometric_accuracy=0.9,
            readability_score=0.8,
            cambridge_compliance=0.85,
            label_placement_score=0.7,
            collision_detection_score=0.9,
            overall_quality=0.0,  # Will be calculated
        )

        # Check that overall quality was calculated
        expected_overall = 0.9 * 0.3 + 0.8 * 0.25 + 0.85 * 0.25 + 0.7 * 0.1 + 0.9 * 0.1
        assert abs(validation.overall_quality - expected_overall) < 0.001

    def test_validation_with_issues(self):
        """Test validation with quality issues"""
        validation = DiagramValidationResult(
            geometric_accuracy=0.6,
            readability_score=0.5,
            cambridge_compliance=0.7,
            label_placement_score=0.4,
            collision_detection_score=0.3,
            overall_quality=0.5,
            validation_issues=[
                "Labels overlap with diagram elements",
                "Angle measurements unclear",
            ],
            improvement_suggestions=[
                "Increase label buffer distance",
                "Use larger font for angle values",
            ],
        )

        assert len(validation.validation_issues) == 2
        assert len(validation.improvement_suggestions) == 2
        assert "Labels overlap" in validation.validation_issues[0]


class TestDiagramGenerationWorkflow:
    """Test complete diagram generation workflow models"""

    def test_generation_request(self):
        """Test diagram generation request"""
        request = DiagramGenerationRequest(
            question_text="Triangle ABC has AB = 10 cm and angle BAC = 60°. Find the area.",
            diagram_type="static",
            include_labels=True,
            include_measurements=True,
        )

        assert "Triangle ABC" in request.question_text
        assert request.diagram_type == "static"
        assert request.include_labels is True
        assert isinstance(request.canvas_config, DiagramCanvas)

    def test_complete_generation_result(self):
        """Test complete diagram generation result"""
        # Create minimal components
        request = DiagramGenerationRequest(question_text="Test triangle")

        manifest = ManifestConstraints(question_id="q1")

        solution = GeometricSolution()

        code = ManifestDiagramCode(manim_code="test code", scene_class_name="TestScene")

        validation = DiagramValidationResult(
            geometric_accuracy=0.9,
            readability_score=0.8,
            cambridge_compliance=0.85,
            label_placement_score=0.7,
            collision_detection_score=0.9,
            overall_quality=0.83,
        )

        result = DiagramGenerationResult(
            request=request,
            manifest_constraints=manifest,
            geometric_solution=solution,
            manim_code=code,
            validation_result=validation,
            processing_time=23.5,
            success=True,
            quality_passed=True,
        )

        assert result.success is True
        assert result.quality_passed is True
        assert result.processing_time == 23.5
        assert isinstance(result.generation_id, str)
        assert isinstance(result.created_at, datetime)


class TestModelIntegration:
    """Test integration between different model components"""

    def test_cgv_pipeline_data_flow(self):
        """Test data flows correctly through CGV pipeline models"""
        # 1. CONSTRAIN: Extract constraints from text
        constraint = GeometricConstraint(
            constraint_id="c1",
            constraint_type=GeometricConstraintType.EQUALITY,
            target_elements=["side_AB"],
            value=10.0,
            text_source="AB = 10 cm",
        )

        triangle = ShapeDefinition(
            shape_id="tri1", shape_type=ShapeType.TRIANGLE, vertices=["A", "B", "C"]
        )

        manifest = ManifestConstraints(
            question_id="q1", shapes=[triangle], constraints=[constraint], unknowns=["side_BC"]
        )

        # 2. GENERATE: Solve constraints
        solution = GeometricSolution(
            solved_points={
                "A": CoordinatePoint(x=0, y=0),
                "B": CoordinatePoint(x=10, y=0),
                "C": CoordinatePoint(x=5, y=8.66),
            },
            solved_values={"side_BC": 10.0},
            is_valid=True,
        )

        # 3. VERIFY: Generate and validate diagram
        code = ManifestDiagramCode(
            manim_code="# Generated Manim code here", scene_class_name="TriangleDiagram"
        )

        validation = DiagramValidationResult(
            geometric_accuracy=0.95,
            readability_score=0.9,
            cambridge_compliance=0.85,
            label_placement_score=0.8,
            collision_detection_score=0.9,
            overall_quality=0.88,
        )

        # Verify data flows correctly
        assert manifest.constraints[0].value == solution.solved_points["B"].x
        assert validation.overall_quality > 0.8
        assert solution.is_valid is True
