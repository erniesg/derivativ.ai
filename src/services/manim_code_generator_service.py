"""
Manim Code Generator Service - GENERATE DIAGRAM step of CGV pipeline.
Converts geometric solutions to Manim code for diagram rendering.
Integrates with E2B for secure code execution and validation.
"""

import logging
import uuid

from src.models.diagram_models import (
    CoordinatePoint,
    GeometricSolution,
    ManifestConstraints,
    ManifestDiagramCode,
    ShapeDefinition,
    ShapeType,
)

logger = logging.getLogger(__name__)


class ManimCodeGeneratorService:
    """
    Service for generating Manim code from geometric solutions.
    Creates production-ready Manim scenes for IGCSE mathematics diagrams.
    """

    def __init__(self):
        self.manim_version = "0.18.0"
        self.scene_counter = 0

        # Manim styling for IGCSE compliance
        self.igcse_colors = {
            "line": "BLACK",
            "point": "BLACK",
            "label": "BLACK",
            "angle_arc": "BLUE",
            "measurement": "RED",
            "background": "WHITE",
        }

        self.igcse_fonts = {"label_size": 24, "measurement_size": 20, "title_size": 28}

    async def generate_manim_code(
        self, manifest_constraints: ManifestConstraints, geometric_solution: GeometricSolution
    ) -> ManifestDiagramCode:
        """
        Generate complete Manim code from geometric solution.

        Args:
            manifest_constraints: Original constraints and shapes
            geometric_solution: Solved coordinates and values

        Returns:
            ManifestDiagramCode: Complete Manim scene code
        """
        logger.info(f"Generating Manim code for question {manifest_constraints.question_id}")

        try:
            # Generate unique scene class name
            scene_name = self._generate_scene_name()

            # Build Manim code components
            imports = self._generate_imports()
            scene_class = self._generate_scene_class(
                scene_name, manifest_constraints, geometric_solution
            )

            # Combine into complete code
            manim_code = f"{imports}\n\n{scene_class}"

            # Calculate complexity and render time
            complexity = self._calculate_complexity(manifest_constraints, geometric_solution)
            render_time = self._estimate_render_time(complexity)

            result = ManifestDiagramCode(
                manim_code=manim_code,
                scene_class_name=scene_name,
                complexity_score=complexity,
                estimated_render_time=render_time,
                dependencies=["manim", "numpy", "math"],
            )

            logger.info(f"Generated Manim code with complexity {complexity:.2f}")
            return result

        except Exception as e:
            logger.error(f"Manim code generation failed: {e}")
            # Return fallback code
            return self._generate_fallback_code(manifest_constraints.question_id)

    def _generate_scene_name(self) -> str:
        """Generate unique scene class name"""
        self.scene_counter += 1
        return f"DiagramScene_{self.scene_counter}_{uuid.uuid4().hex[:8]}"

    def _generate_imports(self) -> str:
        """Generate Manim imports"""
        return """from manim import *
import numpy as np
import math"""

    def _generate_scene_class(
        self,
        scene_name: str,
        manifest_constraints: ManifestConstraints,
        geometric_solution: GeometricSolution,
    ) -> str:
        """Generate complete Manim scene class"""

        # Extract canvas configuration
        canvas = geometric_solution.canvas_bounds

        # Generate scene header
        scene_header = f"""class {scene_name}(Scene):
    def construct(self):
        # IGCSE Mathematics Diagram
        self.camera.background_color = {self.igcse_colors["background"]}

        # Canvas setup
        canvas_width = {canvas.width}
        canvas_height = {canvas.height}
        scale_factor = {canvas.scale_factor}
        margin = {canvas.margin}"""

        # Generate shape creation code
        shapes_code = self._generate_shapes_code(
            manifest_constraints.shapes, geometric_solution.solved_points
        )

        # Generate measurement annotations
        measurements_code = self._generate_measurements_code(
            manifest_constraints, geometric_solution
        )

        # Generate labels
        labels_code = self._generate_labels_code(geometric_solution.solved_points)

        # Generate positioning and display
        display_code = self._generate_display_code()

        # Combine all parts
        scene_body = f"""

        # Create geometric shapes
{shapes_code}

        # Add measurements and annotations
{measurements_code}

        # Add vertex labels
{labels_code}

        # Position and display diagram
{display_code}"""

        return scene_header + scene_body

    def _generate_shapes_code(
        self, shapes: list[ShapeDefinition], solved_points: dict[str, CoordinatePoint]
    ) -> str:
        """Generate Manim code for geometric shapes"""

        if not shapes:
            return "        # No shapes to render"

        code_lines = []
        shape_objects = []

        for shape in shapes:
            if shape.shape_type == ShapeType.TRIANGLE:
                triangle_code, triangle_obj = self._generate_triangle_code(shape, solved_points)
                code_lines.append(triangle_code)
                shape_objects.append(triangle_obj)

            elif shape.shape_type == ShapeType.QUADRILATERAL:
                quad_code, quad_obj = self._generate_quadrilateral_code(shape, solved_points)
                code_lines.append(quad_code)
                shape_objects.append(quad_obj)

            elif shape.shape_type == ShapeType.LINE:
                line_code, line_obj = self._generate_line_code(shape, solved_points)
                code_lines.append(line_code)
                shape_objects.append(line_obj)

            elif shape.shape_type == ShapeType.CIRCLE:
                circle_code, circle_obj = self._generate_circle_code(shape, solved_points)
                code_lines.append(circle_code)
                shape_objects.append(circle_obj)

        # Group all shapes
        if shape_objects:
            code_lines.append(f"        diagram_shapes = VGroup({', '.join(shape_objects)})")

        return "\n".join(code_lines)

    def _generate_triangle_code(
        self, shape: ShapeDefinition, solved_points: dict[str, CoordinatePoint]
    ) -> tuple[str, str]:
        """Generate Manim code for triangle"""

        if len(shape.vertices) != 3:
            return "        # Invalid triangle", "None"

        a, b, c = shape.vertices
        obj_name = f"triangle_{shape.shape_id}"

        # Get coordinates
        try:
            pa = solved_points[a]
            pb = solved_points[b]
            pc = solved_points[c]
        except KeyError as e:
            logger.warning(f"Missing vertex {e} for triangle {shape.shape_id}")
            return f"        # Missing vertex {e}", "None"

        code = f"""        # Triangle {a}{b}{c}
        point_{a} = np.array([{pa.x:.3f}, {pa.y:.3f}, 0])
        point_{b} = np.array([{pb.x:.3f}, {pb.y:.3f}, 0])
        point_{c} = np.array([{pc.x:.3f}, {pc.y:.3f}, 0])
        {obj_name} = Polygon(
            point_{a}, point_{b}, point_{c},
            color={self.igcse_colors["line"]},
            stroke_width=2,
            fill_opacity=0
        )"""

        return code, obj_name

    def _generate_quadrilateral_code(
        self, shape: ShapeDefinition, solved_points: dict[str, CoordinatePoint]
    ) -> tuple[str, str]:
        """Generate Manim code for quadrilateral"""

        if len(shape.vertices) != 4:
            return "        # Invalid quadrilateral", "None"

        a, b, c, d = shape.vertices
        obj_name = f"quad_{shape.shape_id}"

        # Get coordinates
        try:
            pa = solved_points[a]
            pb = solved_points[b]
            pc = solved_points[c]
            pd = solved_points[d]
        except KeyError as e:
            logger.warning(f"Missing vertex {e} for quadrilateral {shape.shape_id}")
            return f"        # Missing vertex {e}", "None"

        code = f"""        # Quadrilateral {a}{b}{c}{d}
        point_{a} = np.array([{pa.x:.3f}, {pa.y:.3f}, 0])
        point_{b} = np.array([{pb.x:.3f}, {pb.y:.3f}, 0])
        point_{c} = np.array([{pc.x:.3f}, {pc.y:.3f}, 0])
        point_{d} = np.array([{pd.x:.3f}, {pd.y:.3f}, 0])
        {obj_name} = Polygon(
            point_{a}, point_{b}, point_{c}, point_{d},
            color={self.igcse_colors["line"]},
            stroke_width=2,
            fill_opacity=0
        )"""

        return code, obj_name

    def _generate_line_code(
        self, shape: ShapeDefinition, solved_points: dict[str, CoordinatePoint]
    ) -> tuple[str, str]:
        """Generate Manim code for line segment"""

        if len(shape.vertices) != 2:
            return "        # Invalid line", "None"

        a, b = shape.vertices
        obj_name = f"line_{shape.shape_id}"

        # Get coordinates
        try:
            pa = solved_points[a]
            pb = solved_points[b]
        except KeyError as e:
            logger.warning(f"Missing vertex {e} for line {shape.shape_id}")
            return f"        # Missing vertex {e}", "None"

        code = f"""        # Line {a}{b}
        point_{a} = np.array([{pa.x:.3f}, {pa.y:.3f}, 0])
        point_{b} = np.array([{pb.x:.3f}, {pb.y:.3f}, 0])
        {obj_name} = Line(
            point_{a}, point_{b},
            color={self.igcse_colors["line"]},
            stroke_width=2
        )"""

        return code, obj_name

    def _generate_circle_code(
        self, shape: ShapeDefinition, solved_points: dict[str, CoordinatePoint]
    ) -> tuple[str, str]:
        """Generate Manim code for circle"""

        if len(shape.vertices) < 1:
            return "        # Invalid circle", "None"

        center_vertex = shape.vertices[0]
        obj_name = f"circle_{shape.shape_id}"

        # Get center coordinates
        try:
            center = solved_points[center_vertex]
        except KeyError as e:
            logger.warning(f"Missing center {e} for circle {shape.shape_id}")
            return f"        # Missing center {e}", "None"

        # Default radius (could be extracted from constraints)
        radius = shape.properties.get("radius", 2.0)

        code = f"""        # Circle centered at {center_vertex}
        center_{center_vertex} = np.array([{center.x:.3f}, {center.y:.3f}, 0])
        {obj_name} = Circle(
            radius={radius},
            color={self.igcse_colors["line"]},
            stroke_width=2,
            fill_opacity=0
        ).move_to(center_{center_vertex})"""

        return code, obj_name

    def _generate_measurements_code(
        self, manifest_constraints: ManifestConstraints, geometric_solution: GeometricSolution
    ) -> str:
        """Generate measurement annotations (lengths, angles)"""

        code_lines = ["        # Measurements and annotations"]
        measurement_objects = []

        # Add length measurements
        for constraint in manifest_constraints.constraints:
            if constraint.constraint_type.value == "equality" and constraint.value:
                if len(constraint.target_elements) == 1:
                    element = constraint.target_elements[0]

                    if element.startswith("side_"):
                        # Side length measurement
                        vertices = element[5:]  # Remove "side_" prefix
                        if len(vertices) == 2:
                            measurement_code, measurement_obj = self._generate_length_measurement(
                                vertices[0], vertices[1], constraint.value, constraint.unit or "cm"
                            )
                            if measurement_obj != "None":
                                code_lines.append(measurement_code)
                                measurement_objects.append(measurement_obj)

            elif (
                constraint.constraint_type.value == "angle"
                and constraint.value
                and len(constraint.target_elements) == 1
            ):
                # Angle measurement
                element = constraint.target_elements[0]
                if element.startswith("angle_") and len(element[6:]) == 3:
                    vertices = element[6:]  # Remove "angle_" prefix
                    angle_code, angle_obj = self._generate_angle_measurement(
                        vertices[0], vertices[1], vertices[2], constraint.value
                    )
                    if angle_obj != "None":
                        code_lines.append(angle_code)
                        measurement_objects.append(angle_obj)

        # Group measurements
        if measurement_objects:
            code_lines.append(f"        measurements = VGroup({', '.join(measurement_objects)})")

        return "\n".join(code_lines)

    def _generate_length_measurement(
        self, vertex1: str, vertex2: str, value: float, unit: str
    ) -> tuple[str, str]:
        """Generate length measurement annotation"""

        obj_name = f"length_{vertex1}{vertex2}"

        code = f"""        # Length measurement {vertex1}{vertex2} = {value} {unit}
        midpoint_{vertex1}{vertex2} = (point_{vertex1} + point_{vertex2}) / 2
        direction_{vertex1}{vertex2} = point_{vertex2} - point_{vertex1}
        normal_{vertex1}{vertex2} = np.array([-direction_{vertex1}{vertex2}[1], direction_{vertex1}{vertex2}[0], 0])
        normal_{vertex1}{vertex2} = normal_{vertex1}{vertex2} / np.linalg.norm(normal_{vertex1}{vertex2}) * 0.3

        {obj_name} = MathTex(
            "{value} \\\\text{{ {unit}}}",
            font_size={self.igcse_fonts["measurement_size"]},
            color={self.igcse_colors["measurement"]}
        ).move_to(midpoint_{vertex1}{vertex2} + normal_{vertex1}{vertex2})"""

        return code, obj_name

    def _generate_angle_measurement(
        self, vertex_a: str, vertex_b: str, vertex_c: str, angle_degrees: float
    ) -> tuple[str, str]:
        """Generate angle measurement annotation (B is the vertex of angle ABC)"""

        obj_name = f"angle_{vertex_a}{vertex_b}{vertex_c}"

        code = f"""        # Angle measurement ∠{vertex_a}{vertex_b}{vertex_c} = {angle_degrees}°
        # Calculate angle arc
        vec_ba = point_{vertex_a} - point_{vertex_b}
        vec_bc = point_{vertex_c} - point_{vertex_b}

        # Normalize vectors for angle calculation
        vec_ba_norm = vec_ba / np.linalg.norm(vec_ba)
        vec_bc_norm = vec_bc / np.linalg.norm(vec_bc)

        # Calculate start and end angles for arc
        start_angle_{vertex_b} = math.atan2(vec_ba_norm[1], vec_ba_norm[0])
        end_angle_{vertex_b} = math.atan2(vec_bc_norm[1], vec_bc_norm[0])

        arc_{obj_name} = Arc(
            radius=0.5,
            start_angle=start_angle_{vertex_b},
            angle=end_angle_{vertex_b} - start_angle_{vertex_b},
            color={self.igcse_colors["angle_arc"]},
            stroke_width=1.5
        ).move_to(point_{vertex_b})

        # Angle label
        label_angle = math.atan2(
            (vec_ba_norm[1] + vec_bc_norm[1]) / 2,
            (vec_ba_norm[0] + vec_bc_norm[0]) / 2
        )
        label_pos = point_{vertex_b} + 0.7 * np.array([math.cos(label_angle), math.sin(label_angle), 0])

        {obj_name} = VGroup(
            arc_{obj_name},
            MathTex(
                "{angle_degrees}°",
                font_size={self.igcse_fonts["measurement_size"]},
                color={self.igcse_colors["angle_arc"]}
            ).move_to(label_pos)
        )"""

        return code, obj_name

    def _generate_labels_code(self, solved_points: dict[str, CoordinatePoint]) -> str:
        """Generate vertex labels"""

        code_lines = ["        # Vertex labels"]
        label_objects = []

        for vertex, point in solved_points.items():
            obj_name = f"label_{vertex}"

            # Position label slightly offset from point
            offset_x = 0.3 if point.x >= 0 else -0.3
            offset_y = 0.3 if point.y >= 0 else -0.3

            code = f"""        {obj_name} = Text(
            "{vertex}",
            font_size={self.igcse_fonts["label_size"]},
            color={self.igcse_colors["label"]}
        ).move_to(np.array([{point.x:.3f} + {offset_x}, {point.y:.3f} + {offset_y}, 0]))"""

            code_lines.append(code)
            label_objects.append(obj_name)

        # Group labels
        if label_objects:
            code_lines.append(f"        vertex_labels = VGroup({', '.join(label_objects)})")

        return "\n".join(code_lines)

    def _generate_display_code(self) -> str:
        """Generate final display and positioning code"""

        return """
        # Collect all diagram elements
        diagram_elements = VGroup()
        if 'diagram_shapes' in locals():
            diagram_elements.add(diagram_shapes)
        if 'measurements' in locals():
            diagram_elements.add(measurements)
        if 'vertex_labels' in locals():
            diagram_elements.add(vertex_labels)

        # Center and scale the diagram
        diagram_elements.move_to(ORIGIN)

        # Add "NOT TO SCALE" annotation for IGCSE compliance
        not_to_scale = Text(
            "NOT TO SCALE",
            font_size=20,
            color=BLACK
        ).to_corner(DR)

        # Display everything
        self.add(diagram_elements)
        self.add(not_to_scale)
        self.wait(0.5)"""

    def _calculate_complexity(
        self, manifest_constraints: ManifestConstraints, geometric_solution: GeometricSolution
    ) -> float:
        """Calculate diagram complexity score (0-1)"""

        complexity = 0.0

        # Base complexity from shapes
        shape_complexity = {
            ShapeType.POINT: 0.1,
            ShapeType.LINE: 0.2,
            ShapeType.TRIANGLE: 0.4,
            ShapeType.QUADRILATERAL: 0.6,
            ShapeType.CIRCLE: 0.5,
            ShapeType.POLYGON: 0.8,
        }

        for shape in manifest_constraints.shapes:
            complexity += shape_complexity.get(shape.shape_type, 0.3)

        # Additional complexity from measurements
        complexity += len(manifest_constraints.constraints) * 0.1

        # Additional complexity from solved values
        complexity += len(geometric_solution.solved_values) * 0.05

        # Normalize to 0-1 range
        return min(complexity, 1.0)

    def _estimate_render_time(self, complexity: float) -> float:
        """Estimate rendering time based on complexity"""

        # Base render time + complexity factor
        base_time = 5.0  # seconds
        complexity_factor = complexity * 20.0

        return base_time + complexity_factor

    def _generate_fallback_code(self, question_id: str) -> ManifestDiagramCode:
        """Generate simple fallback Manim code when generation fails"""

        scene_name = f"FallbackScene_{uuid.uuid4().hex[:8]}"

        fallback_code = f"""from manim import *
import numpy as np

class {scene_name}(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        # Fallback diagram - simple triangle
        triangle = Polygon(
            np.array([-2, -1, 0]),
            np.array([2, -1, 0]),
            np.array([0, 2, 0]),
            color=BLACK,
            stroke_width=2,
            fill_opacity=0
        )

        # Labels
        label_a = Text("A", font_size=24, color=BLACK).next_to(triangle.get_vertices()[0], DL)
        label_b = Text("B", font_size=24, color=BLACK).next_to(triangle.get_vertices()[1], DR)
        label_c = Text("C", font_size=24, color=BLACK).next_to(triangle.get_vertices()[2], UP)

        # Error message
        error_text = Text(
            "Diagram generation failed - fallback diagram",
            font_size=16,
            color=RED
        ).to_edge(DOWN)

        self.add(triangle, label_a, label_b, label_c, error_text)
        self.wait(0.5)"""

        return ManifestDiagramCode(
            manim_code=fallback_code,
            scene_class_name=scene_name,
            complexity_score=0.3,
            estimated_render_time=10.0,
        )
