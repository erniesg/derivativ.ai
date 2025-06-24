"""
Unit tests for Manim code generator service.
Tests the GENERATE DIAGRAM step of CGV pipeline.
"""


import pytest

from src.models.diagram_models import (
    CoordinatePoint,
    DiagramCanvas,
    GeometricConstraint,
    GeometricConstraintType,
    GeometricSolution,
    ManifestConstraints,
    ShapeDefinition,
    ShapeType,
)
from src.services.manim_code_generator_service import ManimCodeGeneratorService


class TestManimCodeGeneratorService:
    """Test Manim code generation from geometric solutions"""

    @pytest.fixture
    def code_generator(self):
        """Manim code generator service for testing"""
        return ManimCodeGeneratorService()

    async def test_generate_triangle_code(self, code_generator):
        """Test generating Manim code for triangle"""

        # Create triangle with solved coordinates
        triangle = ShapeDefinition(
            shape_id="tri1", shape_type=ShapeType.TRIANGLE, vertices=["A", "B", "C"]
        )

        constraints = [
            GeometricConstraint(
                constraint_id="c1",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=["side_AB"],
                value=5.0,
                unit="cm",
                text_source="AB = 5 cm",
            )
        ]

        manifest = ManifestConstraints(
            question_id="test_triangle", shapes=[triangle], constraints=constraints
        )

        solution = GeometricSolution(
            solved_points={
                "A": CoordinatePoint(x=0, y=0, label="A"),
                "B": CoordinatePoint(x=5, y=0, label="B"),
                "C": CoordinatePoint(x=2.5, y=4.33, label="C"),
            },
            is_valid=True,
        )

        # Generate code
        result = await code_generator.generate_manim_code(manifest, solution)

        # Verify result structure
        assert result.manim_code is not None
        assert len(result.manim_code) > 0
        assert result.scene_class_name.startswith("DiagramScene_")
        assert 0 <= result.complexity_score <= 1.0
        assert result.estimated_render_time > 0

        # Verify code content
        code = result.manim_code
        assert "from manim import *" in code
        assert "class " + result.scene_class_name in code
        assert "def construct(self):" in code
        assert "background_color = WHITE" in code

        # Check triangle creation
        assert "Polygon(" in code
        assert "point_A" in code
        assert "point_B" in code
        assert "point_C" in code

        # Check coordinates are included
        assert "0.000" in code  # Point A coordinates
        assert "5.000" in code  # Point B coordinates
        assert "4.330" in code  # Point C y-coordinate (approximately)

    async def test_generate_quadrilateral_code(self, code_generator):
        """Test generating Manim code for quadrilateral"""

        quad = ShapeDefinition(
            shape_id="quad1", shape_type=ShapeType.QUADRILATERAL, vertices=["P", "Q", "R", "S"]
        )

        manifest = ManifestConstraints(question_id="test_quad", shapes=[quad])

        solution = GeometricSolution(
            solved_points={
                "P": CoordinatePoint(x=0, y=0, label="P"),
                "Q": CoordinatePoint(x=4, y=0, label="Q"),
                "R": CoordinatePoint(x=4, y=3, label="R"),
                "S": CoordinatePoint(x=0, y=3, label="S"),
            },
            is_valid=True,
        )

        result = await code_generator.generate_manim_code(manifest, solution)

        # Verify quadrilateral-specific code
        code = result.manim_code
        assert "quad_quad1" in code
        assert "point_P" in code
        assert "point_Q" in code
        assert "point_R" in code
        assert "point_S" in code

        # Check all four vertices are used in polygon
        assert "point_P, point_Q, point_R, point_S" in code

    async def test_generate_line_code(self, code_generator):
        """Test generating Manim code for line segment"""

        line = ShapeDefinition(shape_id="line1", shape_type=ShapeType.LINE, vertices=["A", "B"])

        manifest = ManifestConstraints(question_id="test_line", shapes=[line])

        solution = GeometricSolution(
            solved_points={
                "A": CoordinatePoint(x=-2, y=1, label="A"),
                "B": CoordinatePoint(x=3, y=-1, label="B"),
            },
            is_valid=True,
        )

        result = await code_generator.generate_manim_code(manifest, solution)

        # Verify line-specific code
        code = result.manim_code
        assert "Line(" in code
        assert "point_A, point_B" in code
        assert "-2.000" in code  # Point A x-coordinate
        assert "3.000" in code  # Point B x-coordinate

    async def test_generate_measurements(self, code_generator):
        """Test generating measurement annotations"""

        triangle = ShapeDefinition(
            shape_id="tri1", shape_type=ShapeType.TRIANGLE, vertices=["A", "B", "C"]
        )

        constraints = [
            GeometricConstraint(
                constraint_id="c1",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=["side_AB"],
                value=10.0,
                unit="cm",
                text_source="AB = 10 cm",
            ),
            GeometricConstraint(
                constraint_id="c2",
                constraint_type=GeometricConstraintType.ANGLE,
                target_elements=["angle_ABC"],
                value=60.0,
                unit="degrees",
                text_source="angle ABC = 60°",
            ),
        ]

        manifest = ManifestConstraints(
            question_id="test_measurements", shapes=[triangle], constraints=constraints
        )

        solution = GeometricSolution(
            solved_points={
                "A": CoordinatePoint(x=0, y=0, label="A"),
                "B": CoordinatePoint(x=10, y=0, label="B"),
                "C": CoordinatePoint(x=5, y=8.66, label="C"),
            },
            is_valid=True,
        )

        result = await code_generator.generate_manim_code(manifest, solution)

        # Verify measurement annotations
        code = result.manim_code

        # Check length measurement
        assert "10 \\\\text{ cm}" in code or "10" in code
        assert "length_AB" in code
        assert "midpoint_AB" in code

        # Check angle measurement
        assert "60°" in code or "60" in code
        assert "angle_ABC" in code
        assert "Arc(" in code
        assert "start_angle" in code

    async def test_generate_labels(self, code_generator):
        """Test generating vertex labels"""

        triangle = ShapeDefinition(
            shape_id="tri1", shape_type=ShapeType.TRIANGLE, vertices=["A", "B", "C"]
        )

        manifest = ManifestConstraints(question_id="test_labels", shapes=[triangle])

        solution = GeometricSolution(
            solved_points={
                "A": CoordinatePoint(x=0, y=0, label="A"),
                "B": CoordinatePoint(x=3, y=0, label="B"),
                "C": CoordinatePoint(x=1.5, y=2.6, label="C"),
            },
            is_valid=True,
        )

        result = await code_generator.generate_manim_code(manifest, solution)

        # Verify label generation
        code = result.manim_code

        # Check all vertex labels
        assert 'Text(\n            "A"' in code
        assert 'Text(\n            "B"' in code
        assert 'Text(\n            "C"' in code

        # Check label positioning
        assert "label_A" in code
        assert "label_B" in code
        assert "label_C" in code
        assert "vertex_labels" in code

    async def test_igcse_compliance_features(self, code_generator):
        """Test IGCSE compliance features in generated code"""

        triangle = ShapeDefinition(
            shape_id="tri1", shape_type=ShapeType.TRIANGLE, vertices=["A", "B", "C"]
        )

        manifest = ManifestConstraints(question_id="test_igcse", shapes=[triangle])

        solution = GeometricSolution(
            solved_points={
                "A": CoordinatePoint(x=0, y=0),
                "B": CoordinatePoint(x=1, y=0),
                "C": CoordinatePoint(x=0.5, y=0.866),
            },
            is_valid=True,
        )

        result = await code_generator.generate_manim_code(manifest, solution)

        # Verify IGCSE compliance
        code = result.manim_code

        # Check background color
        assert "background_color = WHITE" in code

        # Check "NOT TO SCALE" annotation
        assert "NOT TO SCALE" in code
        assert "to_corner(DR)" in code

        # Check color scheme
        assert "color=BLACK" in code

        # Check font sizes are reasonable
        assert "font_size=24" in code or "font_size=20" in code

    async def test_complexity_calculation(self, code_generator):
        """Test complexity score calculation"""

        # Simple triangle
        simple_triangle = ShapeDefinition(
            shape_id="simple", shape_type=ShapeType.TRIANGLE, vertices=["A", "B", "C"]
        )

        simple_manifest = ManifestConstraints(
            question_id="simple", shapes=[simple_triangle], constraints=[]
        )

        simple_solution = GeometricSolution(
            solved_points={
                "A": CoordinatePoint(x=0, y=0),
                "B": CoordinatePoint(x=1, y=0),
                "C": CoordinatePoint(x=0.5, y=1),
            }
        )

        simple_result = await code_generator.generate_manim_code(simple_manifest, simple_solution)

        # Complex diagram with multiple shapes and constraints
        complex_shapes = [
            ShapeDefinition(
                shape_id="tri1", shape_type=ShapeType.TRIANGLE, vertices=["A", "B", "C"]
            ),
            ShapeDefinition(
                shape_id="quad1", shape_type=ShapeType.QUADRILATERAL, vertices=["D", "E", "F", "G"]
            ),
            ShapeDefinition(shape_id="line1", shape_type=ShapeType.LINE, vertices=["H", "I"]),
        ]

        complex_constraints = [
            GeometricConstraint(
                constraint_id="c1",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=["side_AB"],
                value=5.0,
                text_source="AB = 5",
            ),
            GeometricConstraint(
                constraint_id="c2",
                constraint_type=GeometricConstraintType.ANGLE,
                target_elements=["angle_ABC"],
                value=60.0,
                text_source="angle ABC = 60°",
            ),
        ]

        complex_manifest = ManifestConstraints(
            question_id="complex", shapes=complex_shapes, constraints=complex_constraints
        )

        complex_solution = GeometricSolution(
            solved_points={
                vertex: CoordinatePoint(x=i, y=i % 2)
                for i, vertex in enumerate(["A", "B", "C", "D", "E", "F", "G", "H", "I"])
            },
            solved_values={"x": 5.0, "y": 10.0},
        )

        complex_result = await code_generator.generate_manim_code(
            complex_manifest, complex_solution
        )

        # Complex diagram should have higher complexity
        assert complex_result.complexity_score > simple_result.complexity_score
        assert complex_result.estimated_render_time > simple_result.estimated_render_time

    async def test_missing_coordinates_handling(self, code_generator):
        """Test handling of missing coordinate data"""

        triangle = ShapeDefinition(
            shape_id="incomplete", shape_type=ShapeType.TRIANGLE, vertices=["A", "B", "C"]
        )

        manifest = ManifestConstraints(question_id="incomplete", shapes=[triangle])

        # Missing coordinate for vertex C
        incomplete_solution = GeometricSolution(
            solved_points={
                "A": CoordinatePoint(x=0, y=0),
                "B": CoordinatePoint(x=1, y=0),
                # Missing "C"
            },
            is_valid=False,
        )

        result = await code_generator.generate_manim_code(manifest, incomplete_solution)

        # Should still generate code with fallback or handle gracefully
        assert result.manim_code is not None
        assert len(result.manim_code) > 0

        # Check for error handling comments
        code = result.manim_code
        assert "Missing vertex" in code or "point_A" in code

    async def test_fallback_code_generation(self, code_generator):
        """Test fallback code generation when main generation fails"""

        # Test fallback generation directly
        fallback = code_generator._generate_fallback_code("test_fallback")

        # Verify fallback code structure
        assert fallback.manim_code is not None
        assert "FallbackScene_" in fallback.scene_class_name
        assert "fallback diagram" in fallback.manim_code.lower()
        assert "from manim import *" in fallback.manim_code
        assert "class " in fallback.manim_code
        assert fallback.complexity_score == 0.3
        assert fallback.estimated_render_time == 10.0

    async def test_canvas_bounds_integration(self, code_generator):
        """Test integration with canvas bounds from geometric solution"""

        triangle = ShapeDefinition(
            shape_id="canvas_test", shape_type=ShapeType.TRIANGLE, vertices=["A", "B", "C"]
        )

        manifest = ManifestConstraints(question_id="canvas_test", shapes=[triangle])

        custom_canvas = DiagramCanvas(width=20.0, height=15.0, scale_factor=1.5, margin=2.0)

        solution = GeometricSolution(
            solved_points={
                "A": CoordinatePoint(x=0, y=0),
                "B": CoordinatePoint(x=5, y=0),
                "C": CoordinatePoint(x=2.5, y=4),
            },
            canvas_bounds=custom_canvas,
            is_valid=True,
        )

        result = await code_generator.generate_manim_code(manifest, solution)

        # Verify canvas configuration is used
        code = result.manim_code
        assert "canvas_width = 20.0" in code
        assert "canvas_height = 15.0" in code
        assert "scale_factor = 1.5" in code
        assert "margin = 2.0" in code

    def test_scene_name_generation(self, code_generator):
        """Test unique scene name generation"""

        name1 = code_generator._generate_scene_name()
        name2 = code_generator._generate_scene_name()

        # Should be unique
        assert name1 != name2
        assert name1.startswith("DiagramScene_")
        assert name2.startswith("DiagramScene_")

        # Should contain counter
        assert "DiagramScene_1_" in name1
        assert "DiagramScene_2_" in name2

    def test_igcse_color_and_font_constants(self, code_generator):
        """Test IGCSE styling constants"""

        # Verify color scheme
        assert code_generator.igcse_colors["background"] == "WHITE"
        assert code_generator.igcse_colors["line"] == "BLACK"
        assert code_generator.igcse_colors["label"] == "BLACK"

        # Verify font sizes
        assert 16 <= code_generator.igcse_fonts["label_size"] <= 32
        assert 16 <= code_generator.igcse_fonts["measurement_size"] <= 32
        assert code_generator.igcse_fonts["title_size"] >= 24
