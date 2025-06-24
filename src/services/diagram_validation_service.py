"""
Diagram Validation Service - VERIFY step of CGV pipeline.
Validates generated Manim diagrams for quality, accuracy, and IGCSE compliance.
Integrates with E2B for secure code execution and rendering validation.
"""

import ast
import logging
import math
import re
from typing import Any

from src.models.diagram_models import (
    DiagramValidationResult,
    GeometricSolution,
    ManifestConstraints,
    ManifestDiagramCode,
)

logger = logging.getLogger(__name__)


class DiagramValidationService:
    """
    Service for validating generated Manim diagrams.
    Ensures geometric accuracy, readability, and Cambridge IGCSE compliance.
    """

    def __init__(self):
        # Quality thresholds for validation
        self.quality_thresholds = {
            "geometric_accuracy": 0.85,
            "readability_score": 0.80,
            "cambridge_compliance": 0.90,
            "label_placement": 0.75,
            "collision_detection": 0.80,
            "overall_minimum": 0.80,
        }

        # IGCSE compliance requirements
        self.igcse_requirements = {
            "must_have_not_to_scale": True,
            "background_color": "WHITE",
            "text_color": "BLACK",
            "line_color": "BLACK",
            "minimum_font_size": 16,
            "maximum_font_size": 32,
        }

    async def validate_diagram(
        self,
        manim_code: ManifestDiagramCode,
        manifest_constraints: ManifestConstraints,
        geometric_solution: GeometricSolution,
    ) -> DiagramValidationResult:
        """
        Comprehensive validation of generated Manim diagram.

        Args:
            manim_code: Generated Manim code to validate
            manifest_constraints: Original constraints from text
            geometric_solution: Solved coordinates

        Returns:
            DiagramValidationResult: Detailed validation assessment
        """
        logger.info(f"Validating Manim diagram for scene {manim_code.scene_class_name}")

        try:
            # Parse and analyze Manim code
            code_analysis = self._analyze_manim_code(manim_code.manim_code)

            # Validate geometric accuracy
            geometric_accuracy = self._validate_geometric_accuracy(
                code_analysis, manifest_constraints, geometric_solution
            )

            # Validate readability and visual clarity
            readability_score = self._validate_readability(code_analysis, manim_code)

            # Validate Cambridge IGCSE compliance
            cambridge_compliance = self._validate_cambridge_compliance(code_analysis)

            # Validate label placement
            label_placement_score = self._validate_label_placement(
                code_analysis, geometric_solution
            )

            # Detect visual collisions and overlaps
            collision_detection_score = self._validate_collision_detection(
                code_analysis, geometric_solution
            )

            # Calculate overall quality score
            overall_quality = self._calculate_overall_quality(
                geometric_accuracy,
                readability_score,
                cambridge_compliance,
                label_placement_score,
                collision_detection_score,
            )

            # Generate validation issues and suggestions
            issues = self._identify_issues(
                geometric_accuracy,
                readability_score,
                cambridge_compliance,
                label_placement_score,
                collision_detection_score,
                code_analysis=code_analysis,
            )

            suggestions = self._generate_improvement_suggestions(issues, code_analysis)

            result = DiagramValidationResult(
                geometric_accuracy=geometric_accuracy,
                readability_score=readability_score,
                cambridge_compliance=cambridge_compliance,
                label_placement_score=label_placement_score,
                collision_detection_score=collision_detection_score,
                overall_quality=overall_quality,
                validation_issues=issues,
                improvement_suggestions=suggestions,
            )

            logger.info(f"Validation complete: overall quality {overall_quality:.3f}")
            return result

        except Exception as e:
            logger.error(f"Diagram validation failed: {e}")
            return self._generate_error_validation_result(str(e))

    def _analyze_manim_code(self, manim_code: str) -> dict[str, Any]:
        """Parse and analyze Manim code structure"""

        analysis = {
            "has_imports": False,
            "scene_class_found": False,
            "construct_method_found": False,
            "background_color": None,
            "shapes": [],
            "labels": [],
            "measurements": [],
            "colors_used": [],
            "font_sizes": [],
            "has_not_to_scale": False,
            "coordinate_points": {},
            "syntax_valid": True,
            "estimated_elements": 0,
        }

        try:
            # Check syntax validity
            ast.parse(manim_code)
            analysis["syntax_valid"] = True
        except SyntaxError as e:
            logger.warning(f"Syntax error in Manim code: {e}")
            analysis["syntax_valid"] = False
            return analysis

        # Analyze code structure
        lines = manim_code.split("\n")

        for line in lines:
            line_clean = line.strip()

            # Check imports
            if line_clean.startswith("from manim import") or line_clean.startswith("import manim"):
                analysis["has_imports"] = True

            # Check scene class
            if "class" in line_clean and "Scene" in line_clean:
                analysis["scene_class_found"] = True

            # Check construct method
            if "def construct(self):" in line_clean:
                analysis["construct_method_found"] = True

            # Check background color
            if "background_color" in line_clean:
                if "WHITE" in line_clean:
                    analysis["background_color"] = "WHITE"
                elif "BLACK" in line_clean:
                    analysis["background_color"] = "BLACK"

            # Detect shapes
            shape_patterns = [
                r"Polygon\(",
                r"Triangle\(",
                r"Line\(",
                r"Circle\(",
                r"Rectangle\(",
                r"Square\(",
            ]

            for pattern in shape_patterns:
                if re.search(pattern, line_clean):
                    analysis["shapes"].append(pattern.replace("(", "").replace("\\", ""))

            # Detect labels and text
            if re.search(r"Text\(|MathTex\(|Tex\(", line_clean):
                analysis["labels"].append(line_clean)

            # Detect measurements
            if any(
                keyword in line_clean.lower() for keyword in ["cm", "mm", "degree", "°", "angle"]
            ):
                analysis["measurements"].append(line_clean)

            # Extract colors
            color_matches = re.findall(r"color\s*=\s*([A-Z_]+)", line_clean)
            analysis["colors_used"].extend(color_matches)

            # Extract font sizes
            font_size_matches = re.findall(r"font_size\s*=\s*(\d+)", line_clean)
            analysis["font_sizes"].extend([int(size) for size in font_size_matches])

            # Check for "NOT TO SCALE"
            if "NOT TO SCALE" in line_clean:
                analysis["has_not_to_scale"] = True

            # Extract coordinate points
            point_matches = re.findall(r"point_([A-Z])\s*=\s*np\.array\(\[([^]]+)\]\)", line_clean)
            for vertex, coords in point_matches:
                try:
                    coord_values = [float(x.strip()) for x in coords.split(",")[:2]]
                    analysis["coordinate_points"][vertex] = coord_values
                except ValueError:
                    continue

        # Count estimated elements
        analysis["estimated_elements"] = (
            len(analysis["shapes"]) + len(analysis["labels"]) + len(analysis["measurements"])
        )

        return analysis

    def _validate_geometric_accuracy(
        self,
        code_analysis: dict[str, Any],
        manifest_constraints: ManifestConstraints,
        geometric_solution: GeometricSolution,
    ) -> float:
        """Validate geometric accuracy of the diagram"""

        score = 1.0
        issues = []

        # Check if solved coordinates match code coordinates
        code_points = code_analysis["coordinate_points"]
        solved_points = geometric_solution.solved_points

        coordinate_accuracy = 0.0
        if code_points and solved_points:
            matching_points = 0
            total_comparisons = 0

            for vertex in solved_points:
                if vertex in code_points:
                    solved_coords = [solved_points[vertex].x, solved_points[vertex].y]
                    code_coords = code_points[vertex]

                    # Calculate coordinate difference
                    diff = math.sqrt(
                        (solved_coords[0] - code_coords[0]) ** 2
                        + (solved_coords[1] - code_coords[1]) ** 2
                    )

                    # Allow small tolerance for floating point differences
                    if diff < 0.1:
                        matching_points += 1

                    total_comparisons += 1

            if total_comparisons > 0:
                coordinate_accuracy = matching_points / total_comparisons
            else:
                coordinate_accuracy = 0.0

        score *= coordinate_accuracy

        # Check if all required shapes are present
        required_shapes = len(manifest_constraints.shapes)
        generated_shapes = len(code_analysis["shapes"])

        if required_shapes > 0:
            shape_completeness = min(generated_shapes / required_shapes, 1.0)
            score *= shape_completeness

        # Validate constraint representation
        constraint_score = self._validate_constraint_representation(
            code_analysis, manifest_constraints
        )
        score *= constraint_score

        return min(score, 1.0)

    def _validate_readability(
        self, code_analysis: dict[str, Any], manim_code: ManifestDiagramCode
    ) -> float:
        """Validate visual readability and clarity"""

        score = 1.0

        # Check font sizes are appropriate
        font_sizes = code_analysis["font_sizes"]
        if font_sizes:
            appropriate_sizes = [
                size
                for size in font_sizes
                if self.igcse_requirements["minimum_font_size"]
                <= size
                <= self.igcse_requirements["maximum_font_size"]
            ]
            font_score = len(appropriate_sizes) / len(font_sizes)
            score *= font_score

        # Check color contrast
        colors = code_analysis["colors_used"]
        if colors:
            # Prefer BLACK on WHITE for IGCSE
            good_colors = ["BLACK", "BLUE", "RED"]  # High contrast colors
            contrast_score = len([c for c in colors if c in good_colors]) / len(colors)
            score *= contrast_score

        # Check element density (avoid overcrowding)
        elements = code_analysis["estimated_elements"]
        if elements > 10:
            density_penalty = 1.0 - min((elements - 10) * 0.05, 0.3)
            score *= density_penalty

        # Check code complexity vs estimated render time
        complexity = manim_code.complexity_score
        render_time = manim_code.estimated_render_time

        # Penalize if render time is too high for complexity
        if render_time > complexity * 50:  # Reasonable threshold
            efficiency_score = complexity * 50 / render_time
            score *= efficiency_score

        return min(score, 1.0)

    def _validate_cambridge_compliance(self, code_analysis: dict[str, Any]) -> float:
        """Validate Cambridge IGCSE compliance requirements"""

        score = 1.0
        compliance_checks = []

        # Check background color
        if code_analysis["background_color"] == self.igcse_requirements["background_color"]:
            compliance_checks.append(1.0)
        else:
            compliance_checks.append(0.0)

        # Check for "NOT TO SCALE" annotation
        if code_analysis["has_not_to_scale"] == self.igcse_requirements["must_have_not_to_scale"]:
            compliance_checks.append(1.0)
        else:
            compliance_checks.append(0.5)  # Partial credit if missing

        # Check color scheme (prefer BLACK lines on WHITE background)
        colors = code_analysis["colors_used"]
        if colors:
            black_usage = colors.count("BLACK") / len(colors)
            compliance_checks.append(black_usage)
        else:
            compliance_checks.append(0.8)  # Default colors are usually okay

        # Check font sizes are within range
        font_sizes = code_analysis["font_sizes"]
        if font_sizes:
            valid_sizes = [
                1.0
                if self.igcse_requirements["minimum_font_size"]
                <= size
                <= self.igcse_requirements["maximum_font_size"]
                else 0.0
                for size in font_sizes
            ]
            font_compliance = sum(valid_sizes) / len(valid_sizes)
            compliance_checks.append(font_compliance)
        else:
            compliance_checks.append(0.9)  # No fonts detected, assume defaults

        # Calculate average compliance
        if compliance_checks:
            score = sum(compliance_checks) / len(compliance_checks)

        return min(score, 1.0)

    def _validate_label_placement(
        self, code_analysis: dict[str, Any], geometric_solution: GeometricSolution
    ) -> float:
        """Validate label positioning and readability"""

        score = 1.0

        # Check if labels exist for all vertices
        solved_vertices = set(geometric_solution.solved_points.keys())
        code_points = set(code_analysis["coordinate_points"].keys())

        if solved_vertices:
            label_coverage = len(code_points) / len(solved_vertices)
            score *= label_coverage

        # Check label positioning (look for offset calculations)
        label_lines = code_analysis["labels"]
        positioning_quality = 0.0

        if label_lines:
            good_positioning = 0
            for line in label_lines:
                # Look for proper offset or positioning
                if any(keyword in line for keyword in ["move_to", "next_to", "offset", "+", "-"]):
                    good_positioning += 1

            positioning_quality = good_positioning / len(label_lines)

        score *= positioning_quality

        return min(score, 1.0)

    def _validate_collision_detection(
        self, code_analysis: dict[str, Any], geometric_solution: GeometricSolution
    ) -> float:
        """Detect potential visual collisions and overlaps"""

        score = 1.0

        # Check for coordinate clustering (points too close together)
        code_points = code_analysis["coordinate_points"]
        if len(code_points) > 1:
            min_distance = float("inf")
            vertices = list(code_points.keys())

            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    v1, v2 = vertices[i], vertices[j]
                    coords1 = code_points[v1]
                    coords2 = code_points[v2]

                    distance = math.sqrt(
                        (coords1[0] - coords2[0]) ** 2 + (coords1[1] - coords2[1]) ** 2
                    )
                    min_distance = min(min_distance, distance)

            # Penalize if minimum distance is too small
            if min_distance < 0.5:  # Minimum readable distance
                collision_penalty = min_distance / 0.5
                score *= collision_penalty

        # Check for overlapping text (simple heuristic)
        label_count = len(code_analysis["labels"])
        if label_count > len(code_points):
            # More labels than points might indicate overlapping
            overlap_penalty = len(code_points) / label_count
            score *= overlap_penalty

        return min(score, 1.0)

    def _validate_constraint_representation(
        self, code_analysis: dict[str, Any], manifest_constraints: ManifestConstraints
    ) -> float:
        """Validate that constraints are properly represented in the diagram"""

        score = 1.0

        # Count how many constraints have visual representation
        represented_constraints = 0
        total_constraints = len(manifest_constraints.constraints)

        for constraint in manifest_constraints.constraints:
            constraint_represented = False

            if constraint.constraint_type.value == "equality" and constraint.value:
                # Look for measurement annotations in code
                for measurement_line in code_analysis["measurements"]:
                    if str(constraint.value) in measurement_line:
                        constraint_represented = True
                        break

            elif constraint.constraint_type.value == "angle":
                # Look for angle annotations
                for measurement_line in code_analysis["measurements"]:
                    if "°" in measurement_line or "degree" in measurement_line.lower():
                        constraint_represented = True
                        break

            if constraint_represented:
                represented_constraints += 1

        if total_constraints > 0:
            representation_score = represented_constraints / total_constraints
            score *= representation_score

        return min(score, 1.0)

    def _calculate_overall_quality(self, *scores: float) -> float:
        """Calculate weighted overall quality score"""

        weights = [
            0.3,
            0.25,
            0.25,
            0.1,
            0.1,
        ]  # Geometric, Readability, Cambridge, Labels, Collisions

        if len(scores) != len(weights):
            # Fallback to simple average
            return sum(scores) / len(scores)

        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        return min(weighted_sum, 1.0)

    def _identify_issues(self, *scores: float, code_analysis: dict[str, Any]) -> list[str]:
        """Identify specific validation issues"""

        issues = []
        score_names = [
            "geometric_accuracy",
            "readability_score",
            "cambridge_compliance",
            "label_placement_score",
            "collision_detection_score",
        ]

        # Check individual scores against thresholds
        for score, name in zip(scores, score_names):
            threshold = self.quality_thresholds.get(name, 0.8)
            if score < threshold:
                issues.append(f"Low {name.replace('_', ' ')}: {score:.2f} < {threshold}")

        # Specific code analysis issues
        if not code_analysis["syntax_valid"]:
            issues.append("Syntax errors detected in generated Manim code")

        if not code_analysis["has_not_to_scale"]:
            issues.append("Missing 'NOT TO SCALE' annotation required for IGCSE")

        if code_analysis["background_color"] != "WHITE":
            issues.append(f"Non-standard background color: {code_analysis['background_color']}")

        if not code_analysis["has_imports"]:
            issues.append("Missing required Manim imports")

        font_sizes = code_analysis["font_sizes"]
        if font_sizes:
            invalid_sizes = [s for s in font_sizes if s < 16 or s > 32]
            if invalid_sizes:
                issues.append(f"Font sizes outside IGCSE range (16-32): {invalid_sizes}")

        return issues

    def _generate_improvement_suggestions(
        self, issues: list[str], code_analysis: dict[str, Any]
    ) -> list[str]:
        """Generate actionable improvement suggestions"""

        suggestions = []

        # Suggestions based on issues
        for issue in issues:
            if "geometric_accuracy" in issue.lower():
                suggestions.append("Verify coordinate calculations match SymPy solution")
                suggestions.append("Check shape vertex ordering and positioning")

            elif "readability" in issue.lower():
                suggestions.append("Increase font sizes for better visibility")
                suggestions.append("Use higher contrast colors (BLACK on WHITE)")
                suggestions.append("Reduce diagram element density")

            elif "cambridge_compliance" in issue.lower():
                suggestions.append("Add 'NOT TO SCALE' annotation in bottom-right corner")
                suggestions.append("Use WHITE background with BLACK lines and text")
                suggestions.append("Ensure font sizes are between 16-32 points")

            elif "label_placement" in issue.lower():
                suggestions.append("Add proper offset to vertex labels")
                suggestions.append("Use next_to() or move_to() for label positioning")
                suggestions.append("Ensure all vertices have visible labels")

            elif "collision" in issue.lower():
                suggestions.append("Increase spacing between diagram elements")
                suggestions.append("Adjust label positioning to avoid overlaps")
                suggestions.append("Scale diagram to reduce crowding")

        # General suggestions based on code analysis
        if not code_analysis["measurements"]:
            suggestions.append("Add measurement annotations for given values")

        if len(code_analysis["shapes"]) == 0:
            suggestions.append("Ensure all geometric shapes are properly rendered")

        if not code_analysis["labels"]:
            suggestions.append("Add vertex labels for all points")

        return list(set(suggestions))  # Remove duplicates

    def _generate_error_validation_result(self, error_message: str) -> DiagramValidationResult:
        """Generate validation result for failed validation"""

        return DiagramValidationResult(
            geometric_accuracy=0.0,
            readability_score=0.0,
            cambridge_compliance=0.0,
            label_placement_score=0.0,
            collision_detection_score=0.0,
            overall_quality=0.0,
            validation_issues=[f"Validation failed: {error_message}"],
            improvement_suggestions=[
                "Check Manim code syntax and structure",
                "Ensure all required components are generated",
                "Verify geometric solution is valid",
            ],
        )

    def quality_passes_threshold(self, validation_result: DiagramValidationResult) -> bool:
        """Check if diagram quality meets minimum thresholds"""

        return (
            validation_result.overall_quality >= self.quality_thresholds["overall_minimum"]
            and validation_result.geometric_accuracy
            >= self.quality_thresholds["geometric_accuracy"]
            and validation_result.cambridge_compliance
            >= self.quality_thresholds["cambridge_compliance"]
        )
