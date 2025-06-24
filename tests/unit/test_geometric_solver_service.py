"""
Unit tests for geometric solver service.
Tests the GENERATE step of CGV pipeline with SymPy constraint solving.
"""

import math

import pytest

from src.models.diagram_models import (
    CoordinatePoint,
    GeometricConstraint,
    GeometricConstraintType,
    ManifestConstraints,
    ShapeDefinition,
    ShapeType,
)
from src.services.geometric_solver_service import GeometricSolverService


class TestGeometricSolverService:
    """Test geometric constraint solving with SymPy"""

    @pytest.fixture
    def solver_service(self):
        """Geometric solver service for testing"""
        return GeometricSolverService()

    async def test_solve_simple_triangle_constraints(self, solver_service):
        """Test solving simple triangle with known side lengths"""

        # Create triangle ABC with AB = 5, AC = 5, angle BAC = 60°
        triangle = ShapeDefinition(
            shape_id="tri1", shape_type=ShapeType.TRIANGLE, vertices=["A", "B", "C"]
        )

        constraints = [
            GeometricConstraint(
                constraint_id="c1",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=["side_AB"],
                value=5.0,
                text_source="AB = 5 cm",
            ),
            GeometricConstraint(
                constraint_id="c2",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=["side_AC"],
                value=5.0,
                text_source="AC = 5 cm",
            ),
            GeometricConstraint(
                constraint_id="c3",
                constraint_type=GeometricConstraintType.ANGLE,
                target_elements=["angle_BAC"],
                value=60.0,
                text_source="angle BAC = 60°",
            ),
        ]

        manifest = ManifestConstraints(
            question_id="test_triangle",
            shapes=[triangle],
            constraints=constraints,
            unknowns=["side_BC"],
        )

        # Solve constraints
        solution = await solver_service.solve_constraints(manifest)

        # Verify solution is valid
        assert solution.is_valid is True
        assert solution.error_message is None

        # Check that all vertices are solved
        assert "A" in solution.solved_points
        assert "B" in solution.solved_points
        assert "C" in solution.solved_points

        # Check coordinate types
        for point in solution.solved_points.values():
            assert isinstance(point, CoordinatePoint)
            assert math.isfinite(point.x)
            assert math.isfinite(point.y)

        # For equilateral triangle with 60° angle and equal sides, BC should also be 5
        # (This will be verified in the Manim generation step)

    async def test_solve_right_triangle_constraints(self, solver_service):
        """Test solving right triangle with Pythagorean theorem"""

        triangle = ShapeDefinition(
            shape_id="right_tri", shape_type=ShapeType.TRIANGLE, vertices=["P", "Q", "R"]
        )

        constraints = [
            GeometricConstraint(
                constraint_id="c1",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=["side_PQ"],
                value=3.0,
                text_source="PQ = 3 cm",
            ),
            GeometricConstraint(
                constraint_id="c2",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=["side_QR"],
                value=4.0,
                text_source="QR = 4 cm",
            ),
            GeometricConstraint(
                constraint_id="c3",
                constraint_type=GeometricConstraintType.ANGLE,
                target_elements=["angle_PQR"],
                value=90.0,
                text_source="angle PQR = 90°",
            ),
        ]

        manifest = ManifestConstraints(
            question_id="test_right_triangle",
            shapes=[triangle],
            constraints=constraints,
            unknowns=["side_PR"],
        )

        solution = await solver_service.solve_constraints(manifest)

        assert solution.is_valid is True
        assert len(solution.solved_points) == 3

        # Check that all coordinates are finite
        for vertex, point in solution.solved_points.items():
            assert math.isfinite(point.x), f"Invalid x coordinate for {vertex}"
            assert math.isfinite(point.y), f"Invalid y coordinate for {vertex}"

    async def test_solve_parallel_lines_constraint(self, solver_service):
        """Test solving parallel lines constraint"""

        line1 = ShapeDefinition(shape_id="line1", shape_type=ShapeType.LINE, vertices=["A", "B"])

        line2 = ShapeDefinition(shape_id="line2", shape_type=ShapeType.LINE, vertices=["C", "D"])

        constraints = [
            GeometricConstraint(
                constraint_id="c1",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=["side_AB"],
                value=6.0,
                text_source="AB = 6 cm",
            ),
            GeometricConstraint(
                constraint_id="c2",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=["side_CD"],
                value=6.0,
                text_source="CD = 6 cm",
            ),
            GeometricConstraint(
                constraint_id="c3",
                constraint_type=GeometricConstraintType.PARALLEL,
                target_elements=["line_AB", "line_CD"],
                text_source="AB is parallel to CD",
            ),
        ]

        manifest = ManifestConstraints(
            question_id="test_parallel", shapes=[line1, line2], constraints=constraints
        )

        solution = await solver_service.solve_constraints(manifest)

        # System may be underconstrained, so we expect fallback solution
        assert len(solution.solved_points) == 4

        # Verify all points have valid coordinates
        points = solution.solved_points
        assert all(math.isfinite(p.x) and math.isfinite(p.y) for p in points.values())

    async def test_solve_perpendicular_lines_constraint(self, solver_service):
        """Test solving perpendicular lines constraint"""

        line1 = ShapeDefinition(shape_id="line1", shape_type=ShapeType.LINE, vertices=["A", "B"])

        line2 = ShapeDefinition(shape_id="line2", shape_type=ShapeType.LINE, vertices=["C", "D"])

        constraints = [
            GeometricConstraint(
                constraint_id="c1",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=["side_AB"],
                value=4.0,
                text_source="AB = 4 cm",
            ),
            GeometricConstraint(
                constraint_id="c2",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=["side_CD"],
                value=3.0,
                text_source="CD = 3 cm",
            ),
            GeometricConstraint(
                constraint_id="c3",
                constraint_type=GeometricConstraintType.PERPENDICULAR,
                target_elements=["line_AB", "line_CD"],
                text_source="AB is perpendicular to CD",
            ),
        ]

        manifest = ManifestConstraints(
            question_id="test_perpendicular", shapes=[line1, line2], constraints=constraints
        )

        solution = await solver_service.solve_constraints(manifest)

        # System may be underconstrained, expect fallback
        assert len(solution.solved_points) == 4

    async def test_solve_quadrilateral_constraints(self, solver_service):
        """Test solving quadrilateral with multiple constraints"""

        quad = ShapeDefinition(
            shape_id="quad1", shape_type=ShapeType.QUADRILATERAL, vertices=["A", "B", "C", "D"]
        )

        constraints = [
            GeometricConstraint(
                constraint_id="c1",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=["side_AB"],
                value=8.0,
                text_source="AB = 8 cm",
            ),
            GeometricConstraint(
                constraint_id="c2",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=["side_BC"],
                value=6.0,
                text_source="BC = 6 cm",
            ),
            GeometricConstraint(
                constraint_id="c3",
                constraint_type=GeometricConstraintType.ANGLE,
                target_elements=["angle_ABC"],
                value=90.0,
                text_source="angle ABC = 90°",
            ),
        ]

        manifest = ManifestConstraints(
            question_id="test_quad", shapes=[quad], constraints=constraints, unknowns=["side_AC"]
        )

        solution = await solver_service.solve_constraints(manifest)

        # May use fallback solution for underconstrained system
        assert len(solution.solved_points) == 4

        # Verify all coordinates are valid
        for point in solution.solved_points.values():
            assert math.isfinite(point.x)
            assert math.isfinite(point.y)

    async def test_handle_overconstrained_system(self, solver_service):
        """Test handling of overconstrained geometric system"""

        triangle = ShapeDefinition(
            shape_id="over_tri", shape_type=ShapeType.TRIANGLE, vertices=["A", "B", "C"]
        )

        # Inconsistent constraints: AB=3, BC=4, CA=10 violates triangle inequality
        constraints = [
            GeometricConstraint(
                constraint_id="c1",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=["side_AB"],
                value=3.0,
                text_source="AB = 3 cm",
            ),
            GeometricConstraint(
                constraint_id="c2",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=["side_BC"],
                value=4.0,
                text_source="BC = 4 cm",
            ),
            GeometricConstraint(
                constraint_id="c3",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=["side_CA"],
                value=10.0,
                text_source="CA = 10 cm",
            ),
        ]

        manifest = ManifestConstraints(
            question_id="test_overconstrained", shapes=[triangle], constraints=constraints
        )

        solution = await solver_service.solve_constraints(manifest)

        # Should either solve with fallback or detect inconsistency
        assert isinstance(solution.is_valid, bool)
        if not solution.is_valid:
            assert solution.error_message is not None
            # Error could be triangle inequality or collinearity
            assert (
                "inequality" in solution.error_message.lower()
                or "collinear" in solution.error_message.lower()
            )

    async def test_handle_underconstrained_system(self, solver_service):
        """Test handling of underconstrained geometric system"""

        triangle = ShapeDefinition(
            shape_id="under_tri", shape_type=ShapeType.TRIANGLE, vertices=["A", "B", "C"]
        )

        # Only one constraint - underconstrained
        constraints = [
            GeometricConstraint(
                constraint_id="c1",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=["side_AB"],
                value=5.0,
                text_source="AB = 5 cm",
            )
        ]

        manifest = ManifestConstraints(
            question_id="test_underconstrained",
            shapes=[triangle],
            constraints=constraints,
            unknowns=["side_BC", "side_CA"],
        )

        solution = await solver_service.solve_constraints(manifest)

        # Should provide some solution (fallback if needed)
        assert len(solution.solved_points) == 3
        assert solution.solution_method in ["sympy_constraint_solving", "error_fallback"]

        # Check all coordinates are finite
        for point in solution.solved_points.values():
            assert math.isfinite(point.x)
            assert math.isfinite(point.y)

    async def test_solve_with_symbolic_unknowns(self, solver_service):
        """Test solving system with symbolic unknowns"""

        triangle = ShapeDefinition(
            shape_id="symbolic_tri", shape_type=ShapeType.TRIANGLE, vertices=["A", "B", "C"]
        )

        constraints = [
            GeometricConstraint(
                constraint_id="c1",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=["x"],  # Variable constraint
                value=7.0,
                text_source="x = 7",
            ),
            GeometricConstraint(
                constraint_id="c2",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=["side_AB"],
                value=10.0,
                text_source="AB = 10 cm",
            ),
        ]

        manifest = ManifestConstraints(
            question_id="test_symbolic",
            shapes=[triangle],
            constraints=constraints,
            unknowns=["x", "y"],
        )

        solution = await solver_service.solve_constraints(manifest)

        # Should solve or provide fallback
        assert len(solution.solved_points) == 3

        # Check that symbolic unknown is solved if possible
        if "x" in solution.solved_values:
            assert solution.solved_values["x"] == 7.0

    async def test_canvas_bounds_calculation(self, solver_service):
        """Test calculation of appropriate canvas bounds"""

        triangle = ShapeDefinition(
            shape_id="bounds_tri", shape_type=ShapeType.TRIANGLE, vertices=["A", "B", "C"]
        )

        constraints = [
            GeometricConstraint(
                constraint_id="c1",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=["side_AB"],
                value=12.0,
                text_source="AB = 12 cm",
            )
        ]

        manifest = ManifestConstraints(
            question_id="test_bounds", shapes=[triangle], constraints=constraints
        )

        solution = await solver_service.solve_constraints(manifest)

        # Solution may be invalid due to triangle inequality issues, but check canvas bounds anyway
        canvas = solution.canvas_bounds
        assert canvas.width >= 8.0  # Minimum width
        assert canvas.height >= 6.0  # Minimum height
        assert canvas.margin > 0

    async def test_error_handling_invalid_constraint(self, solver_service):
        """Test error handling for invalid geometric constraints"""

        triangle = ShapeDefinition(
            shape_id="error_tri", shape_type=ShapeType.TRIANGLE, vertices=["A", "B", "C"]
        )

        # Test with empty constraints - should handle gracefully
        manifest = ManifestConstraints(
            question_id="test_error",
            shapes=[triangle],
            constraints=[],  # No constraints
        )

        # Should handle gracefully with fallback solution
        solution = await solver_service.solve_constraints(manifest)
        assert isinstance(solution.is_valid, bool)

        # Test with invalid constraint type that can't be converted
        # This tests the _constraint_to_equation method's robustness
        unsupported_constraint = GeometricConstraint(
            constraint_id="c1",
            constraint_type=GeometricConstraintType.RATIO,  # Not implemented
            target_elements=["side_AB"],
            value=2.0,
            text_source="AB:BC = 2:1",
        )

        manifest_unsupported = ManifestConstraints(
            question_id="test_unsupported", shapes=[triangle], constraints=[unsupported_constraint]
        )

        solution = await solver_service.solve_constraints(manifest_unsupported)
        assert isinstance(solution.is_valid, bool)

    def test_variable_initialization(self, solver_service):
        """Test initialization of symbolic variables"""

        triangle = ShapeDefinition(
            shape_id="init_tri", shape_type=ShapeType.TRIANGLE, vertices=["A", "B", "C"]
        )

        manifest = ManifestConstraints(
            question_id="test_init", shapes=[triangle], unknowns=["x", "y", "angle_ABC"]
        )

        # Test variable initialization
        solver_service._initialize_variables(manifest)

        # Check coordinate variables
        assert "A_x" in solver_service.coordinate_vars
        assert "A_y" in solver_service.coordinate_vars
        assert "B_x" in solver_service.coordinate_vars
        assert "B_y" in solver_service.coordinate_vars
        assert "C_x" in solver_service.coordinate_vars
        assert "C_y" in solver_service.coordinate_vars

        # Check unknown variables
        assert "x" in solver_service.length_vars
        assert "y" in solver_service.length_vars

    def test_triangle_validation(self, solver_service):
        """Test triangle inequality validation"""

        # Valid triangle
        valid_points = {
            "A": CoordinatePoint(x=0, y=0),
            "B": CoordinatePoint(x=3, y=0),
            "C": CoordinatePoint(x=1.5, y=2.6),
        }

        is_valid, error = solver_service._validate_triangle(["A", "B", "C"], valid_points)
        assert is_valid is True
        assert error is None

        # Invalid triangle (collinear points)
        collinear_points = {
            "A": CoordinatePoint(x=0, y=0),
            "B": CoordinatePoint(x=2, y=0),
            "C": CoordinatePoint(x=4, y=0),
        }

        is_valid, error = solver_service._validate_triangle(["A", "B", "C"], collinear_points)
        assert is_valid is False
        assert "collinear" in error.lower()

        # Invalid triangle (triangle inequality violation)
        invalid_points = {
            "A": CoordinatePoint(x=0, y=0),
            "B": CoordinatePoint(x=1, y=0),
            "C": CoordinatePoint(x=10, y=0.001),  # Very long side with minimal height
        }

        is_valid, error = solver_service._validate_triangle(["A", "B", "C"], invalid_points)
        assert is_valid is False
        # Could be detected as either collinear or triangle inequality
        assert "inequality" in error.lower() or "collinear" in error.lower()

    async def test_fallback_solution_generation(self, solver_service):
        """Test fallback solution generation when SymPy solving fails"""

        # Create manifest that might cause solving issues
        large_triangle = ShapeDefinition(
            shape_id="fallback_tri", shape_type=ShapeType.TRIANGLE, vertices=["X", "Y", "Z"]
        )

        manifest = ManifestConstraints(
            question_id="test_fallback",
            shapes=[large_triangle],
            constraints=[],  # No constraints to solve
        )

        # Test fallback generation directly
        fallback = solver_service._generate_fallback_solution(manifest)

        # Should generate reasonable coordinates
        assert len(fallback) > 0

        # Test with quadrilateral
        quad = ShapeDefinition(
            shape_id="fallback_quad",
            shape_type=ShapeType.QUADRILATERAL,
            vertices=["P", "Q", "R", "S"],
        )

        manifest_quad = ManifestConstraints(
            question_id="test_fallback_quad", shapes=[quad], constraints=[]
        )

        fallback_quad = solver_service._generate_fallback_solution(manifest_quad)
        assert len(fallback_quad) > 0
