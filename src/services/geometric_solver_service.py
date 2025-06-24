"""
Geometric Solver Service - GENERATE step of CGV pipeline.
Uses SymPy to deterministically solve geometric constraints and produce exact coordinates.
"""

import logging
import math
from typing import Optional

import sympy as sp
from sympy import Eq, cos, solve, sqrt, symbols

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

logger = logging.getLogger(__name__)


class GeometricSolverService:
    """
    Service for solving geometric constraints using SymPy.
    Converts constraint manifests into coordinate solutions deterministically.
    """

    def __init__(self):
        # Standard coordinate variables
        self.coordinate_vars = {}
        self.angle_vars = {}
        self.length_vars = {}

        # Solution cache for performance
        self.solution_cache = {}

    async def solve_constraints(self, manifest: ManifestConstraints) -> GeometricSolution:
        """
        Solve geometric constraints to produce exact coordinate positions.

        Args:
            manifest: Constraints extracted from question text

        Returns:
            GeometricSolution: Solved coordinates and values
        """
        logger.info(f"Solving constraints for question {manifest.question_id}")

        try:
            # Initialize symbolic variables
            self._initialize_variables(manifest)

            # Build constraint equations
            equations = self._build_equations(manifest.constraints)

            # Solve constraint system
            solutions = self._solve_system(equations, manifest)

            # Convert to coordinate points
            solved_points = self._extract_coordinates(solutions, manifest)

            # Calculate derived values
            solved_values = self._calculate_derived_values(solutions, manifest)

            # Validate geometric consistency
            is_valid, error_msg = self._validate_solution(solved_points, manifest)

            # Determine canvas bounds
            canvas_bounds = self._calculate_canvas_bounds(solved_points)

            solution = GeometricSolution(
                solved_points=solved_points,
                solved_values=solved_values,
                is_valid=is_valid,
                error_message=error_msg,
                canvas_bounds=canvas_bounds,
                solution_method="sympy_constraint_solving",
            )

            logger.info(f"Solved {len(solved_points)} points, {len(solved_values)} values")
            return solution

        except Exception as e:
            logger.error(f"Geometric solving failed for question {manifest.question_id}: {e}")
            return GeometricSolution(
                is_valid=False,
                error_message=f"Solving failed: {e!s}",
                solution_method="error_fallback",
            )

    def _initialize_variables(self, manifest: ManifestConstraints):
        """Initialize symbolic variables for all geometric elements"""

        # Clear previous variables
        self.coordinate_vars.clear()
        self.angle_vars.clear()
        self.length_vars.clear()

        # Create coordinate variables for all vertices
        all_vertices = set()
        for shape in manifest.shapes:
            all_vertices.update(shape.vertices)

        for vertex in all_vertices:
            self.coordinate_vars[f"{vertex}_x"] = symbols(f"{vertex}_x", real=True)
            self.coordinate_vars[f"{vertex}_y"] = symbols(f"{vertex}_y", real=True)

        # Create variables for unknowns
        for unknown in manifest.unknowns:
            if unknown not in self.coordinate_vars:
                self.length_vars[unknown] = symbols(unknown, real=True, positive=True)

    def _build_equations(self, constraints: list[GeometricConstraint]) -> list[sp.Eq]:
        """Build SymPy equations from geometric constraints"""

        equations = []

        for constraint in constraints:
            try:
                eq = self._constraint_to_equation(constraint)
                if eq is not None:
                    equations.append(eq)
            except Exception as e:
                logger.warning(
                    f"Failed to convert constraint {constraint.constraint_id} to equation: {e}"
                )
                continue

        return equations

    def _constraint_to_equation(self, constraint: GeometricConstraint) -> Optional[sp.Eq]:
        """Convert a single constraint to a SymPy equation"""

        if constraint.constraint_type == GeometricConstraintType.EQUALITY:
            return self._equality_constraint_to_equation(constraint)
        elif constraint.constraint_type == GeometricConstraintType.ANGLE:
            return self._angle_constraint_to_equation(constraint)
        elif constraint.constraint_type == GeometricConstraintType.DISTANCE:
            return self._distance_constraint_to_equation(constraint)
        elif constraint.constraint_type == GeometricConstraintType.PARALLEL:
            return self._parallel_constraint_to_equation(constraint)
        elif constraint.constraint_type == GeometricConstraintType.PERPENDICULAR:
            return self._perpendicular_constraint_to_equation(constraint)
        else:
            logger.warning(f"Unsupported constraint type: {constraint.constraint_type}")
            return None

    def _equality_constraint_to_equation(self, constraint: GeometricConstraint) -> Optional[sp.Eq]:
        """Convert equality constraint (AB = 10) to SymPy equation"""

        if len(constraint.target_elements) != 1:
            return None

        target = constraint.target_elements[0]
        value = constraint.value

        if target.startswith("side_"):
            # Side length constraint: side_AB = 10
            vertices = target[5:]  # Remove "side_" prefix
            if len(vertices) == 2:
                p1, p2 = vertices[0], vertices[1]

                x1 = self.coordinate_vars.get(f"{p1}_x")
                y1 = self.coordinate_vars.get(f"{p1}_y")
                x2 = self.coordinate_vars.get(f"{p2}_x")
                y2 = self.coordinate_vars.get(f"{p2}_y")

                if all(var is not None for var in [x1, y1, x2, y2]):
                    distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    return Eq(distance, value)

        elif target in self.length_vars:
            # Variable constraint: x = 10
            var = self.length_vars[target]
            return Eq(var, value)

        return None

    def _angle_constraint_to_equation(self, constraint: GeometricConstraint) -> Optional[sp.Eq]:
        """Convert angle constraint (angle ABC = 60°) to SymPy equation"""

        if len(constraint.target_elements) != 1:
            return None

        target = constraint.target_elements[0]
        value_degrees = constraint.value

        if target.startswith("angle_"):
            # Angle constraint: angle_ABC = 60 (B is the vertex)
            vertices = target[6:]  # Remove "angle_" prefix
            if len(vertices) == 3:
                vertex_a, vertex_b, vertex_c = vertices[0], vertices[1], vertices[2]

                # Get coordinates for angle ABC (B is the vertex)
                xa = self.coordinate_vars.get(f"{vertex_a}_x")
                ya = self.coordinate_vars.get(f"{vertex_a}_y")
                xb = self.coordinate_vars.get(f"{vertex_b}_x")
                yb = self.coordinate_vars.get(f"{vertex_b}_y")
                xc = self.coordinate_vars.get(f"{vertex_c}_x")
                yc = self.coordinate_vars.get(f"{vertex_c}_y")

                if all(var is not None for var in [xa, ya, xb, yb, xc, yc]):
                    # Vector BA and BC
                    ba_x, ba_y = xa - xb, ya - yb
                    bc_x, bc_y = xc - xb, yc - yb

                    # Dot product and magnitudes
                    dot_product = ba_x * bc_x + ba_y * bc_y
                    mag_ba = sqrt(ba_x**2 + ba_y**2)
                    mag_bc = sqrt(bc_x**2 + bc_y**2)

                    # cos(angle) = dot_product / (mag_ba * mag_bc)
                    value_radians = sp.rad(value_degrees)
                    cos_angle = cos(value_radians)

                    return Eq(dot_product / (mag_ba * mag_bc), cos_angle)

        return None

    def _distance_constraint_to_equation(self, constraint: GeometricConstraint) -> Optional[sp.Eq]:
        """Convert distance constraint to SymPy equation"""

        if len(constraint.target_elements) != 2:
            return None

        p1_name, p2_name = constraint.target_elements[0], constraint.target_elements[1]
        distance_value = constraint.value

        x1 = self.coordinate_vars.get(f"{p1_name}_x")
        y1 = self.coordinate_vars.get(f"{p1_name}_y")
        x2 = self.coordinate_vars.get(f"{p2_name}_x")
        y2 = self.coordinate_vars.get(f"{p2_name}_y")

        if all(var is not None for var in [x1, y1, x2, y2]):
            distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            return Eq(distance, distance_value)

        return None

    def _parallel_constraint_to_equation(self, constraint: GeometricConstraint) -> Optional[sp.Eq]:
        """Convert parallel constraint to SymPy equation"""

        if len(constraint.target_elements) != 2:
            return None

        line1, line2 = constraint.target_elements[0], constraint.target_elements[1]

        # Extract vertices from line names
        if line1.startswith("line_") and line2.startswith("line_"):
            vertices1 = line1[5:]  # Remove "line_" prefix
            vertices2 = line2[5:]

            if len(vertices1) == 2 and len(vertices2) == 2:
                p1a, p1b = vertices1[0], vertices1[1]
                p2a, p2b = vertices2[0], vertices2[1]

                # Get coordinates
                coords = {}
                for vertex in [p1a, p1b, p2a, p2b]:
                    coords[f"{vertex}_x"] = self.coordinate_vars.get(f"{vertex}_x")
                    coords[f"{vertex}_y"] = self.coordinate_vars.get(f"{vertex}_y")

                if all(var is not None for var in coords.values()):
                    # Direction vectors
                    dir1_x = coords[f"{p1b}_x"] - coords[f"{p1a}_x"]
                    dir1_y = coords[f"{p1b}_y"] - coords[f"{p1a}_y"]
                    dir2_x = coords[f"{p2b}_x"] - coords[f"{p2a}_x"]
                    dir2_y = coords[f"{p2b}_y"] - coords[f"{p2a}_y"]

                    # Parallel means cross product = 0
                    cross_product = dir1_x * dir2_y - dir1_y * dir2_x
                    return Eq(cross_product, 0)

        return None

    def _perpendicular_constraint_to_equation(
        self, constraint: GeometricConstraint
    ) -> Optional[sp.Eq]:
        """Convert perpendicular constraint to SymPy equation"""

        if len(constraint.target_elements) != 2:
            return None

        line1, line2 = constraint.target_elements[0], constraint.target_elements[1]

        # Extract vertices from line names
        if line1.startswith("line_") and line2.startswith("line_"):
            vertices1 = line1[5:]  # Remove "line_" prefix
            vertices2 = line2[5:]

            if len(vertices1) == 2 and len(vertices2) == 2:
                p1a, p1b = vertices1[0], vertices1[1]
                p2a, p2b = vertices2[0], vertices2[1]

                # Get coordinates
                coords = {}
                for vertex in [p1a, p1b, p2a, p2b]:
                    coords[f"{vertex}_x"] = self.coordinate_vars.get(f"{vertex}_x")
                    coords[f"{vertex}_y"] = self.coordinate_vars.get(f"{vertex}_y")

                if all(var is not None for var in coords.values()):
                    # Direction vectors
                    dir1_x = coords[f"{p1b}_x"] - coords[f"{p1a}_x"]
                    dir1_y = coords[f"{p1b}_y"] - coords[f"{p1a}_y"]
                    dir2_x = coords[f"{p2b}_x"] - coords[f"{p2a}_x"]
                    dir2_y = coords[f"{p2b}_y"] - coords[f"{p2a}_y"]

                    # Perpendicular means dot product = 0
                    dot_product = dir1_x * dir2_x + dir1_y * dir2_y
                    return Eq(dot_product, 0)

        return None

    def _solve_system(self, equations: list[sp.Eq], manifest: ManifestConstraints) -> dict:
        """Solve the system of equations using SymPy"""

        if not equations:
            logger.warning("No equations to solve")
            return {}

        # Collect all variables
        all_vars = []
        all_vars.extend(self.coordinate_vars.values())
        all_vars.extend(self.length_vars.values())
        all_vars.extend(self.angle_vars.values())

        # Remove None values
        all_vars = [var for var in all_vars if var is not None]

        if not all_vars:
            logger.warning("No variables to solve for")
            return {}

        logger.info(f"Solving system with {len(equations)} equations and {len(all_vars)} variables")

        try:
            # Add positioning constraints to make system well-defined
            positioned_equations = self._add_positioning_constraints(equations, manifest)

            # Solve the system
            solutions = solve(positioned_equations, all_vars, dict=True)

            if solutions:
                # Return first solution (assuming unique solution)
                return solutions[0]
            else:
                logger.warning("No solutions found for constraint system")
                return {}

        except Exception as e:
            logger.error(f"SymPy solving failed: {e}")
            # Return fallback solutions
            return self._generate_fallback_solution(manifest)

    def _add_positioning_constraints(
        self, equations: list[sp.Eq], manifest: ManifestConstraints
    ) -> list[sp.Eq]:
        """Add positioning constraints to make the system well-defined"""

        positioned_equations = equations.copy()

        # Fix first vertex at origin to eliminate translation freedom
        if manifest.shapes:
            first_shape = manifest.shapes[0]
            if first_shape.vertices:
                first_vertex = first_shape.vertices[0]

                # Set first vertex at origin
                x_var = self.coordinate_vars.get(f"{first_vertex}_x")
                y_var = self.coordinate_vars.get(f"{first_vertex}_y")

                if x_var is not None:
                    positioned_equations.append(Eq(x_var, 0))
                if y_var is not None:
                    positioned_equations.append(Eq(y_var, 0))

                # Fix second vertex on positive x-axis to eliminate rotation freedom
                if len(first_shape.vertices) > 1:
                    second_vertex = first_shape.vertices[1]
                    y_var_2 = self.coordinate_vars.get(f"{second_vertex}_y")

                    if y_var_2 is not None:
                        positioned_equations.append(Eq(y_var_2, 0))

        return positioned_equations

    def _generate_fallback_solution(self, manifest: ManifestConstraints) -> dict:
        """Generate a simple fallback solution when solving fails"""

        logger.info("Generating fallback solution using basic geometry")

        fallback = {}

        # Position shapes using simple geometric rules
        for i, shape in enumerate(manifest.shapes):
            if shape.shape_type == ShapeType.TRIANGLE:
                fallback.update(self._fallback_triangle(shape, i))
            elif shape.shape_type == ShapeType.QUADRILATERAL:
                fallback.update(self._fallback_quadrilateral(shape, i))

        return fallback

    def _fallback_triangle(self, shape: ShapeDefinition, index: int) -> dict:
        """Create fallback solution for triangle"""

        if len(shape.vertices) != 3:
            return {}

        a, b, c = shape.vertices
        offset_x = index * 6  # Separate multiple shapes

        return {
            self.coordinate_vars.get(f"{a}_x", f"{a}_x"): 0 + offset_x,
            self.coordinate_vars.get(f"{a}_y", f"{a}_y"): 0,
            self.coordinate_vars.get(f"{b}_x", f"{b}_x"): 4 + offset_x,
            self.coordinate_vars.get(f"{b}_y", f"{b}_y"): 0,
            self.coordinate_vars.get(f"{c}_x", f"{c}_x"): 2 + offset_x,
            self.coordinate_vars.get(f"{c}_y", f"{c}_y"): 3,
        }

    def _fallback_quadrilateral(self, shape: ShapeDefinition, index: int) -> dict:
        """Create fallback solution for quadrilateral"""

        if len(shape.vertices) != 4:
            return {}

        a, b, c, d = shape.vertices
        offset_x = index * 7  # Separate multiple shapes

        return {
            self.coordinate_vars.get(f"{a}_x", f"{a}_x"): 0 + offset_x,
            self.coordinate_vars.get(f"{a}_y", f"{a}_y"): 0,
            self.coordinate_vars.get(f"{b}_x", f"{b}_x"): 5 + offset_x,
            self.coordinate_vars.get(f"{b}_y", f"{b}_y"): 0,
            self.coordinate_vars.get(f"{c}_x", f"{c}_x"): 5 + offset_x,
            self.coordinate_vars.get(f"{c}_y", f"{c}_y"): 3,
            self.coordinate_vars.get(f"{d}_x", f"{d}_x"): 0 + offset_x,
            self.coordinate_vars.get(f"{d}_y", f"{d}_y"): 3,
        }

    def _extract_coordinates(
        self, solutions: dict, manifest: ManifestConstraints
    ) -> dict[str, CoordinatePoint]:
        """Extract coordinate points from SymPy solutions"""

        solved_points = {}

        # Get all unique vertices
        all_vertices = set()
        for shape in manifest.shapes:
            all_vertices.update(shape.vertices)

        for vertex in all_vertices:
            x_var = self.coordinate_vars.get(f"{vertex}_x")
            y_var = self.coordinate_vars.get(f"{vertex}_y")

            x_val = solutions.get(x_var, 0)
            y_val = solutions.get(y_var, 0)

            # Convert SymPy expressions to float
            try:
                x_float = float(x_val.evalf()) if hasattr(x_val, "evalf") else float(x_val)
                y_float = float(y_val.evalf()) if hasattr(y_val, "evalf") else float(y_val)
            except (ValueError, TypeError):
                x_float, y_float = 0.0, 0.0

            solved_points[vertex] = CoordinatePoint(x=x_float, y=y_float, label=vertex)

        return solved_points

    def _calculate_derived_values(
        self, solutions: dict, manifest: ManifestConstraints
    ) -> dict[str, float]:
        """Calculate derived values (unknowns) from solutions"""

        solved_values = {}

        for unknown in manifest.unknowns:
            if unknown in self.length_vars:
                var = self.length_vars[unknown]
                value = solutions.get(var, 0)

                try:
                    float_value = float(value.evalf()) if hasattr(value, "evalf") else float(value)
                    solved_values[unknown] = float_value
                except (ValueError, TypeError):
                    solved_values[unknown] = 0.0

        return solved_values

    def _validate_solution(
        self, solved_points: dict[str, CoordinatePoint], manifest: ManifestConstraints
    ) -> tuple[bool, Optional[str]]:
        """Validate the geometric solution for consistency"""

        # Check for NaN or infinite coordinates
        for vertex, point in solved_points.items():
            if not (math.isfinite(point.x) and math.isfinite(point.y)):
                return False, f"Invalid coordinates for vertex {vertex}: ({point.x}, {point.y})"

        # Check triangle inequality for triangles
        for shape in manifest.shapes:
            if shape.shape_type == ShapeType.TRIANGLE and len(shape.vertices) == 3:
                is_valid, error = self._validate_triangle(shape.vertices, solved_points)
                if not is_valid:
                    return False, error

        # Check for duplicate points
        positions = [(point.x, point.y) for point in solved_points.values()]
        if len(positions) != len(set(positions)):
            return False, "Duplicate vertex positions detected"

        return True, None

    def _validate_triangle(
        self, vertices: list[str], solved_points: dict[str, CoordinatePoint]
    ) -> tuple[bool, Optional[str]]:
        """Validate triangle inequality and non-collinearity"""

        if len(vertices) != 3:
            return True, None

        a, b, c = vertices
        if not all(v in solved_points for v in [a, b, c]):
            return True, None  # Skip validation if points missing

        pa, pb, pc = solved_points[a], solved_points[b], solved_points[c]

        # Calculate side lengths
        ab = math.sqrt((pb.x - pa.x) ** 2 + (pb.y - pa.y) ** 2)
        bc = math.sqrt((pc.x - pb.x) ** 2 + (pc.y - pb.y) ** 2)
        ca = math.sqrt((pa.x - pc.x) ** 2 + (pa.y - pc.y) ** 2)

        # Check non-collinearity first using cross product
        cross_product = (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x)
        tolerance = 1e-6
        if abs(cross_product) < tolerance:
            return False, f"Vertices {a}{b}{c} are collinear"

        # Check triangle inequality
        if ab + bc <= ca + tolerance or bc + ca <= ab + tolerance or ca + ab <= bc + tolerance:
            return False, f"Triangle inequality violated for triangle {a}{b}{c}"

        return True, None

    def _calculate_canvas_bounds(self, solved_points: dict[str, CoordinatePoint]) -> DiagramCanvas:
        """Calculate appropriate canvas bounds for the solved diagram"""

        if not solved_points:
            return DiagramCanvas()

        # Find bounding box
        x_coords = [point.x for point in solved_points.values()]
        y_coords = [point.y for point in solved_points.values()]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # Add margin
        margin = 1.0
        width = max_x - min_x + 2 * margin
        height = max_y - min_y + 2 * margin

        # Ensure minimum size
        width = max(width, 8.0)
        height = max(height, 6.0)

        return DiagramCanvas(width=width, height=height, margin=margin)
