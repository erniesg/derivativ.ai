"""
Constraint Extraction Service - CONSTRAIN step of CGV pipeline.
Extracts geometric constraints from question text using LLM with structured output.
Implements "Text is Law" principle - ignores visual diagram, only parses text.
"""

import logging
import re
from typing import Optional

from src.models.diagram_models import (
    GeometricConstraint,
    GeometricConstraintType,
    ManifestConstraints,
    ShapeDefinition,
    ShapeType,
)
from src.models.llm_models import LLMRequest
from src.services.json_parser import JSONParser
from src.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class ConstraintExtractionService:
    """
    Service for extracting geometric constraints from question text.
    Uses LLM to parse natural language into structured geometric facts.
    """

    def __init__(self, llm_service: LLMService, json_parser: Optional[JSONParser] = None):
        self.llm_service = llm_service
        self.json_parser = json_parser or JSONParser()

        # Cambridge IGCSE geometric patterns
        self.shape_patterns = {
            r"triangle\s+([A-Z]{3})": ShapeType.TRIANGLE,
            r"quadrilateral\s+([A-Z]{4})": ShapeType.QUADRILATERAL,
            r"rectangle\s+([A-Z]{4})": ShapeType.QUADRILATERAL,
            r"square\s+([A-Z]{4})": ShapeType.QUADRILATERAL,
            r"parallelogram\s+([A-Z]{4})": ShapeType.QUADRILATERAL,
            r"circle\s+([A-Z])": ShapeType.CIRCLE,
            r"line\s+([A-Z]{2})": ShapeType.LINE,
        }

        self.constraint_patterns = {
            # Equality patterns: AB = 10, x = 5
            r"([A-Z]{2}|[a-z])\s*=\s*(\d+(?:\.\d+)?)\s*(?:cm|mm|m)?": GeometricConstraintType.EQUALITY,
            # Angle patterns: angle ABC = 60°, ∠ABC = 60°
            r"(?:angle\s+|∠)([A-Z]{3})\s*=\s*(\d+(?:\.\d+)?)\s*°?": GeometricConstraintType.ANGLE,
            # Parallel patterns: AB || CD, AB is parallel to CD, line AB is parallel to line CD
            r"(?:line\s+)?([A-Z]{2})\s*(?:\|\||is\s+parallel\s+to)\s*(?:line\s+)?([A-Z]{2})": GeometricConstraintType.PARALLEL,
            # Perpendicular patterns: AB ⊥ CD, AB is perpendicular to CD, line AB is perpendicular to line CD
            r"(?:line\s+)?([A-Z]{2})\s*(?:⊥|is\s+perpendicular\s+to)\s*(?:line\s+)?([A-Z]{2})": GeometricConstraintType.PERPENDICULAR,
        }

    async def extract_constraints_from_text(
        self, question_text: str, question_id: str
    ) -> ManifestConstraints:
        """
        Extract geometric constraints from question text using LLM + pattern matching.

        Args:
            question_text: Full question text (IGCSE mathematics)
            question_id: Reference to source question

        Returns:
            ManifestConstraints: Structured representation of all constraints
        """
        logger.info(f"Extracting constraints from question {question_id}")

        try:
            # First pass: Use LLM for structured extraction
            llm_constraints = await self._llm_extract_constraints(question_text)

            # Second pass: Pattern matching for missed constraints
            pattern_constraints = self._pattern_extract_constraints(question_text)

            # Third pass: Shape detection
            shapes = self._detect_shapes(question_text)

            # Fourth pass: Unknown variables
            unknowns = self._detect_unknowns(question_text)

            # Combine and validate
            all_constraints = self._merge_constraints(llm_constraints, pattern_constraints)

            manifest = ManifestConstraints(
                question_id=question_id,
                shapes=shapes,
                constraints=all_constraints,
                unknowns=unknowns,
                extraction_method="llm_plus_patterns",
            )

            logger.info(
                f"Extracted {len(all_constraints)} constraints, {len(shapes)} shapes, {len(unknowns)} unknowns"
            )
            return manifest

        except Exception as e:
            logger.error(f"Constraint extraction failed for question {question_id}: {e}")
            # Return minimal manifest to avoid pipeline failure
            return ManifestConstraints(question_id=question_id, extraction_method="error_fallback")

    async def _llm_extract_constraints(self, question_text: str) -> list[GeometricConstraint]:
        """Use LLM to extract constraints with structured prompting"""

        prompt = f"""
        TASK: Extract ALL geometric constraints from this Cambridge IGCSE Mathematics question text.

        CRITICAL RULES:
        1. IGNORE any visual diagram - extract ONLY from the text
        2. "Text is Law" - if text says angle = 30°, use 30° even if diagram shows 60°
        3. Extract ALL measurements, relationships, and geometric facts
        4. Include implicit constraints (e.g., "triangle ABC" implies 3 vertices)

        QUESTION TEXT:
        {question_text}

        OUTPUT FORMAT (JSON):
        {{
            "constraints": [
                {{
                    "constraint_id": "c1",
                    "constraint_type": "equality|angle|parallel|perpendicular|tangent|radius",
                    "target_elements": ["side_AB", "angle_ABC", "line_AB", etc.],
                    "value": numeric_value_or_null,
                    "unit": "cm|degrees|mm|null",
                    "text_source": "exact text where constraint was found",
                    "confidence": 0.0_to_1.0
                }}
            ]
        }}

        EXAMPLES:
        - "AB = 10 cm" → {{"constraint_type": "equality", "target_elements": ["side_AB"], "value": 10, "unit": "cm"}}
        - "angle ABC = 60°" → {{"constraint_type": "angle", "target_elements": ["angle_ABC"], "value": 60, "unit": "degrees"}}
        - "AB is parallel to CD" → {{"constraint_type": "parallel", "target_elements": ["line_AB", "line_CD"], "value": null}}
        """

        request = LLMRequest(
            model="gpt-4o-mini",  # Use reliable model for constraint extraction
            prompt=prompt,
            temperature=0.1,  # Low temperature for precise extraction
            max_tokens=2000,
            stream=False,
        )

        response = await self.llm_service.generate_non_stream(request)

        # Parse JSON response
        try:
            json_data = self.json_parser.parse_json_from_text(response.content)
            constraints = []

            for i, constraint_data in enumerate(json_data.get("constraints", [])):
                constraint = GeometricConstraint(
                    constraint_id=constraint_data.get("constraint_id", f"llm_c{i+1}"),
                    constraint_type=GeometricConstraintType(constraint_data["constraint_type"]),
                    target_elements=constraint_data["target_elements"],
                    value=constraint_data.get("value"),
                    unit=constraint_data.get("unit"),
                    text_source=constraint_data["text_source"],
                    confidence=constraint_data.get("confidence", 0.8),
                )
                constraints.append(constraint)

            return constraints

        except Exception as e:
            logger.warning(f"LLM constraint extraction parsing failed: {e}")
            return []

    def _pattern_extract_constraints(self, question_text: str) -> list[GeometricConstraint]:
        """Extract constraints using regex patterns as backup"""

        constraints = []
        constraint_id_counter = 1

        for pattern, constraint_type in self.constraint_patterns.items():
            matches = re.finditer(pattern, question_text, re.IGNORECASE)

            for match in matches:
                if constraint_type == GeometricConstraintType.EQUALITY:
                    target = match.group(1)
                    value = float(match.group(2))
                    # Determine if it's a side or variable
                    if len(target) == 2 and target.isupper():
                        target_elements = [f"side_{target}"]
                    else:
                        target_elements = [target]

                    constraint = GeometricConstraint(
                        constraint_id=f"pat_c{constraint_id_counter}",
                        constraint_type=constraint_type,
                        target_elements=target_elements,
                        value=value,
                        unit="cm",  # Default unit for IGCSE
                        text_source=match.group(0),
                        confidence=0.9,
                    )

                elif constraint_type == GeometricConstraintType.ANGLE:
                    angle_name = match.group(1)
                    value = float(match.group(2))

                    constraint = GeometricConstraint(
                        constraint_id=f"pat_c{constraint_id_counter}",
                        constraint_type=constraint_type,
                        target_elements=[f"angle_{angle_name}"],
                        value=value,
                        unit="degrees",
                        text_source=match.group(0),
                        confidence=0.95,
                    )

                elif constraint_type in [
                    GeometricConstraintType.PARALLEL,
                    GeometricConstraintType.PERPENDICULAR,
                ]:
                    line1 = match.group(1)
                    line2 = match.group(2)

                    constraint = GeometricConstraint(
                        constraint_id=f"pat_c{constraint_id_counter}",
                        constraint_type=constraint_type,
                        target_elements=[f"line_{line1}", f"line_{line2}"],
                        text_source=match.group(0),
                        confidence=0.85,
                    )

                constraints.append(constraint)
                constraint_id_counter += 1

        return constraints

    def _detect_shapes(self, question_text: str) -> list[ShapeDefinition]:
        """Detect geometric shapes mentioned in the text"""

        shapes = []
        shape_id_counter = 1

        for pattern, shape_type in self.shape_patterns.items():
            matches = re.finditer(pattern, question_text, re.IGNORECASE)

            for match in matches:
                vertices_str = match.group(1)
                vertices = list(vertices_str)

                shape = ShapeDefinition(
                    shape_id=f"shape_{shape_id_counter}",
                    shape_type=shape_type,
                    vertices=vertices,
                    properties={"detected_from": match.group(0)},
                )
                shapes.append(shape)
                shape_id_counter += 1

        return shapes

    def _detect_unknowns(self, question_text: str) -> list[str]:
        """Detect unknown variables to be solved"""

        unknowns = []

        # Pattern for common unknowns in IGCSE
        unknown_patterns = [
            r"find\s+(?:the\s+)?([a-z])",  # "find x", "find the value of y"
            r"calculate\s+(?:the\s+)?([a-z])",  # "calculate x"
            r"work\s+out\s+(?:the\s+)?([a-z])",  # "work out x"
            r"(?:angle|∠)\s+([A-Z]{3})",  # "angle ABC" (when asking to find)
            r"(?:area|perimeter|volume)\s+of",  # "area of triangle"
        ]

        for pattern in unknown_patterns:
            matches = re.finditer(pattern, question_text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) > 0:
                    unknown = match.group(1)
                    if unknown not in unknowns:
                        unknowns.append(unknown)

        # Add common Cambridge unknowns if question asks to "find"
        if re.search(r"find|calculate|work\s+out", question_text, re.IGNORECASE):
            common_unknowns = ["x", "y", "area", "perimeter", "angle"]
            for unknown in common_unknowns:
                if unknown in question_text.lower() and unknown not in unknowns:
                    unknowns.append(unknown)

        return unknowns

    def _merge_constraints(
        self,
        llm_constraints: list[GeometricConstraint],
        pattern_constraints: list[GeometricConstraint],
    ) -> list[GeometricConstraint]:
        """Merge constraints from LLM and pattern matching, removing duplicates"""

        all_constraints = llm_constraints.copy()

        # Add pattern constraints that aren't already covered by LLM
        for pattern_constraint in pattern_constraints:
            is_duplicate = False

            for llm_constraint in llm_constraints:
                if pattern_constraint.constraint_type == llm_constraint.constraint_type and set(
                    pattern_constraint.target_elements
                ) == set(llm_constraint.target_elements):
                    is_duplicate = True
                    break

            if not is_duplicate:
                all_constraints.append(pattern_constraint)

        return all_constraints

    def validate_constraints(
        self, constraints: list[GeometricConstraint]
    ) -> tuple[bool, list[str]]:
        """
        Validate extracted constraints for geometric consistency.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check for impossible angle values
        for constraint in constraints:
            if (
                constraint.constraint_type == GeometricConstraintType.ANGLE
                and constraint.value
                and (constraint.value <= 0 or constraint.value >= 180)
            ):
                issues.append(
                    f"Invalid angle value: {constraint.value}° in {constraint.text_source}"
                )

        # Check for negative or zero lengths
        for constraint in constraints:
            if (
                constraint.constraint_type == GeometricConstraintType.EQUALITY
                and constraint.value is not None
                and constraint.value <= 0
            ):
                issues.append(
                    f"Invalid length value: {constraint.value} in {constraint.text_source}"
                )

        # Check triangle inequality (basic check)
        triangle_sides = {}
        for constraint in constraints:
            if (
                constraint.constraint_type == GeometricConstraintType.EQUALITY
                and len(constraint.target_elements) == 1
            ):
                element = constraint.target_elements[0]
                if element.startswith("side_") and len(element) == 7:  # side_AB format
                    triangle_name = element[5:]  # Extract "AB"
                    if constraint.value:
                        triangle_sides[triangle_name] = constraint.value

        # Basic triangle inequality check (would need more sophisticated logic for complete validation)
        if len(triangle_sides) >= 3:
            sides = list(triangle_sides.values())
            sides.sort()
            if len(sides) >= 3 and sides[0] + sides[1] <= sides[2]:
                issues.append("Triangle inequality violation detected")

        return len(issues) == 0, issues
