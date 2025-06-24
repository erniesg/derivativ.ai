"""
Unit tests for constraint extraction service.
Tests the CONSTRAIN step of CGV pipeline with focus on "Text is Law" principle.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from src.models.diagram_models import GeometricConstraintType, ShapeType
from src.models.llm_models import LLMResponse
from src.services.constraint_extraction_service import ConstraintExtractionService
from src.services.json_parser import JSONParser
from src.services.llm_service import LLMService


class TestConstraintExtractionService:
    """Test constraint extraction from question text"""

    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service for testing"""
        return AsyncMock(spec=LLMService)

    @pytest.fixture
    def mock_json_parser(self):
        """Mock JSON parser for testing"""
        parser = Mock(spec=JSONParser)
        parser.parse_json_from_text = Mock()
        return parser

    @pytest.fixture
    def extraction_service(self, mock_llm_service, mock_json_parser):
        """Constraint extraction service with mocked dependencies"""
        return ConstraintExtractionService(mock_llm_service, mock_json_parser)

    async def test_extract_triangle_constraints_from_text(
        self, extraction_service, mock_llm_service, mock_json_parser
    ):
        """Test extracting constraints from triangle problem text"""

        question_text = "Triangle ABC has AB = 10 cm, AC = 8 cm and angle BAC = 60°. Find the area of the triangle."

        # Mock LLM response
        llm_response_json = {
            "constraints": [
                {
                    "constraint_id": "c1",
                    "constraint_type": "equality",
                    "target_elements": ["side_AB"],
                    "value": 10,
                    "unit": "cm",
                    "text_source": "AB = 10 cm",
                    "confidence": 0.95,
                },
                {
                    "constraint_id": "c2",
                    "constraint_type": "equality",
                    "target_elements": ["side_AC"],
                    "value": 8,
                    "unit": "cm",
                    "text_source": "AC = 8 cm",
                    "confidence": 0.95,
                },
                {
                    "constraint_id": "c3",
                    "constraint_type": "angle",
                    "target_elements": ["angle_BAC"],
                    "value": 60,
                    "unit": "degrees",
                    "text_source": "angle BAC = 60°",
                    "confidence": 0.9,
                },
            ]
        }

        mock_llm_response = LLMResponse(
            content=f"```json\n{llm_response_json}\n```",
            model_used="gpt-4o-mini",
            provider="openai",
            tokens_used=150,
            cost_estimate=0.01,
            latency_ms=500,
        )
        mock_llm_service.generate_non_stream.return_value = mock_llm_response
        mock_json_parser.parse_json_from_text.return_value = llm_response_json

        # Execute
        result = await extraction_service.extract_constraints_from_text(question_text, "q123")

        # Verify
        assert result.question_id == "q123"
        assert len(result.constraints) >= 3  # LLM + pattern matching may find more

        # Check specific constraints
        equality_constraints = [
            c for c in result.constraints if c.constraint_type == GeometricConstraintType.EQUALITY
        ]
        angle_constraints = [
            c for c in result.constraints if c.constraint_type == GeometricConstraintType.ANGLE
        ]

        assert len(equality_constraints) >= 2  # AB = 10, AC = 8
        assert len(angle_constraints) >= 1  # angle BAC = 60°

        # Verify shapes detected
        triangle_shapes = [s for s in result.shapes if s.shape_type == ShapeType.TRIANGLE]
        assert len(triangle_shapes) >= 1
        assert triangle_shapes[0].vertices == ["A", "B", "C"]

        # Verify unknowns
        assert "area" in result.unknowns

    async def test_extract_angle_measurements_from_text(
        self, extraction_service, mock_llm_service, mock_json_parser
    ):
        """Test extracting angle measurements with different notation"""

        question_text = "In quadrilateral PQRS, angle PQR = 90°, ∠QRS = 120° and angle RSP = 85°. Calculate angle SPQ."

        # Mock empty LLM response to test pattern matching
        mock_llm_response = LLMResponse(
            content='```json\n{"constraints": []}\n```',
            model_used="gpt-4o-mini",
            provider="openai",
            tokens_used=50,
            cost_estimate=0.005,
            latency_ms=300,
        )
        mock_llm_service.generate_non_stream.return_value = mock_llm_response
        mock_json_parser.parse_json_from_text.return_value = {"constraints": []}

        # Execute
        result = await extraction_service.extract_constraints_from_text(question_text, "q456")

        # Verify pattern matching found angles
        angle_constraints = [
            c for c in result.constraints if c.constraint_type == GeometricConstraintType.ANGLE
        ]

        assert len(angle_constraints) >= 3  # Should find all three angles

        # Check specific angle values
        angle_values = {c.target_elements[0]: c.value for c in angle_constraints}

        # Pattern matching should find these angles
        found_angles = set(angle_values.keys())
        expected_angles = {"angle_PQR", "angle_QRS", "angle_RSP"}

        # At least some angles should be found by pattern matching
        assert len(found_angles.intersection(expected_angles)) >= 1

    def test_pattern_extract_equality_constraints(self, extraction_service):
        """Test pattern-based extraction of equality constraints"""

        question_text = "Rectangle ABCD has AB = 12 cm, BC = 8 cm, and x = 5."

        # Test pattern extraction directly
        constraints = extraction_service._pattern_extract_constraints(question_text)

        # Should find AB = 12, BC = 8, x = 5
        equality_constraints = [
            c for c in constraints if c.constraint_type == GeometricConstraintType.EQUALITY
        ]
        assert len(equality_constraints) == 3

        # Check values
        constraint_values = {c.target_elements[0]: c.value for c in equality_constraints}
        assert constraint_values.get("side_AB") == 12.0
        assert constraint_values.get("side_BC") == 8.0
        assert constraint_values.get("x") == 5.0

    def test_pattern_extract_parallel_relationships(self, extraction_service):
        """Test pattern-based extraction of parallel relationships"""

        question_text = "Line AB is parallel to line CD. Also, EF || GH in the diagram."

        constraints = extraction_service._pattern_extract_constraints(question_text)

        parallel_constraints = [
            c for c in constraints if c.constraint_type == GeometricConstraintType.PARALLEL
        ]
        assert len(parallel_constraints) == 2

        # Check first parallel relationship
        first_parallel = parallel_constraints[0]
        assert set(first_parallel.target_elements) == {"line_AB", "line_CD"}

        # Check second parallel relationship
        second_parallel = parallel_constraints[1]
        assert set(second_parallel.target_elements) == {"line_EF", "line_GH"}

    def test_detect_shapes_triangle(self, extraction_service):
        """Test shape detection for triangles"""

        question_text = "Triangle ABC is isosceles. Triangle DEF has a right angle."

        shapes = extraction_service._detect_shapes(question_text)

        triangle_shapes = [s for s in shapes if s.shape_type == ShapeType.TRIANGLE]
        assert len(triangle_shapes) == 2

        # Check vertices
        vertices_sets = [set(s.vertices) for s in triangle_shapes]
        assert {"A", "B", "C"} in vertices_sets
        assert {"D", "E", "F"} in vertices_sets

    def test_detect_shapes_quadrilateral(self, extraction_service):
        """Test shape detection for quadrilaterals"""

        question_text = "Rectangle PQRS has area 48 cm². Square ABCD is shown."

        shapes = extraction_service._detect_shapes(question_text)

        quad_shapes = [s for s in shapes if s.shape_type == ShapeType.QUADRILATERAL]
        assert len(quad_shapes) == 2

        # Check vertices
        vertices_sets = [set(s.vertices) for s in quad_shapes]
        assert {"P", "Q", "R", "S"} in vertices_sets
        assert {"A", "B", "C", "D"} in vertices_sets

    def test_detect_unknowns_find_patterns(self, extraction_service):
        """Test detection of unknown variables from 'find' patterns"""

        question_text = "Find the value of x. Calculate y. Work out the area of triangle."

        unknowns = extraction_service._detect_unknowns(question_text)

        assert "x" in unknowns
        assert "y" in unknowns
        assert "area" in unknowns

    def test_detect_unknowns_cambridge_patterns(self, extraction_service):
        """Test detection of Cambridge IGCSE common unknowns"""

        question_text = "Calculate the perimeter of the shape. Find angle ABC."

        unknowns = extraction_service._detect_unknowns(question_text)

        assert "perimeter" in unknowns
        # Note: angle detection would need more sophisticated pattern matching

    def test_merge_constraints_removes_duplicates(self, extraction_service):
        """Test that merging removes duplicate constraints"""

        from src.models.diagram_models import GeometricConstraint

        # Create duplicate constraints
        constraint1 = GeometricConstraint(
            constraint_id="c1",
            constraint_type=GeometricConstraintType.EQUALITY,
            target_elements=["side_AB"],
            value=10.0,
            text_source="AB = 10",
        )

        constraint2 = GeometricConstraint(
            constraint_id="c2",
            constraint_type=GeometricConstraintType.EQUALITY,
            target_elements=["side_AB"],  # Same target
            value=10.0,
            text_source="AB = 10 cm",
        )

        constraint3 = GeometricConstraint(
            constraint_id="c3",
            constraint_type=GeometricConstraintType.ANGLE,
            target_elements=["angle_ABC"],
            value=60.0,
            text_source="angle ABC = 60°",
        )

        llm_constraints = [constraint1, constraint3]
        pattern_constraints = [constraint2]  # Duplicate of constraint1

        merged = extraction_service._merge_constraints(llm_constraints, pattern_constraints)

        # Should have 2 constraints, not 3 (duplicate removed)
        assert len(merged) == 2

        # Should have both types
        types = [c.constraint_type for c in merged]
        assert GeometricConstraintType.EQUALITY in types
        assert GeometricConstraintType.ANGLE in types

    def test_validate_constraints_valid_cases(self, extraction_service):
        """Test constraint validation for valid cases"""

        from src.models.diagram_models import GeometricConstraint

        valid_constraints = [
            GeometricConstraint(
                constraint_id="c1",
                constraint_type=GeometricConstraintType.ANGLE,
                target_elements=["angle_ABC"],
                value=60.0,  # Valid angle
                text_source="angle ABC = 60°",
            ),
            GeometricConstraint(
                constraint_id="c2",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=["side_AB"],
                value=10.0,  # Valid length
                text_source="AB = 10 cm",
            ),
        ]

        is_valid, issues = extraction_service.validate_constraints(valid_constraints)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_constraints_invalid_angles(self, extraction_service):
        """Test constraint validation catches invalid angles"""

        from src.models.diagram_models import GeometricConstraint

        invalid_constraints = [
            GeometricConstraint(
                constraint_id="c1",
                constraint_type=GeometricConstraintType.ANGLE,
                target_elements=["angle_ABC"],
                value=200.0,  # Invalid angle > 180°
                text_source="angle ABC = 200°",
            ),
            GeometricConstraint(
                constraint_id="c2",
                constraint_type=GeometricConstraintType.ANGLE,
                target_elements=["angle_DEF"],
                value=-30.0,  # Invalid negative angle
                text_source="angle DEF = -30°",
            ),
        ]

        is_valid, issues = extraction_service.validate_constraints(invalid_constraints)

        assert is_valid is False
        assert len(issues) == 2
        assert "Invalid angle value: 200.0°" in issues[0]
        assert "Invalid angle value: -30.0°" in issues[1]

    def test_validate_constraints_invalid_lengths(self, extraction_service):
        """Test constraint validation catches invalid lengths"""

        from src.models.diagram_models import GeometricConstraint

        invalid_constraints = [
            GeometricConstraint(
                constraint_id="c1",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=["side_AB"],
                value=-5.0,  # Invalid negative length
                text_source="AB = -5 cm",
            ),
            GeometricConstraint(
                constraint_id="c2",
                constraint_type=GeometricConstraintType.EQUALITY,
                target_elements=["side_CD"],
                value=0.0,  # Invalid zero length
                text_source="CD = 0 cm",
            ),
        ]

        is_valid, issues = extraction_service.validate_constraints(invalid_constraints)

        assert is_valid is False
        assert len(issues) == 2
        assert "Invalid length value: -5.0" in issues[0]
        assert "Invalid length value: 0.0" in issues[1]

    async def test_extract_constraints_error_handling(
        self, extraction_service, mock_llm_service, mock_json_parser
    ):
        """Test error handling when LLM extraction fails"""

        # Mock LLM service to raise exception
        mock_llm_service.generate_non_stream.side_effect = Exception("LLM service error")

        question_text = "Triangle ABC has AB = 10 cm."

        # Should not raise exception, should return fallback manifest
        result = await extraction_service.extract_constraints_from_text(question_text, "q789")

        assert result.question_id == "q789"
        assert result.extraction_method == "error_fallback"
        # Should still attempt pattern matching even if LLM fails
        assert isinstance(result.constraints, list)

    async def test_text_is_law_principle(
        self, extraction_service, mock_llm_service, mock_json_parser
    ):
        """Test that extraction follows 'Text is Law' principle"""

        # Question where text contradicts typical visual diagram
        question_text = "Triangle ABC appears equilateral in the diagram, but AB = 10 cm, AC = 10 cm, and angle BAC = 30°. NOT TO SCALE."

        # Mock LLM to extract text-based constraints
        llm_response_json = {
            "constraints": [
                {
                    "constraint_id": "c1",
                    "constraint_type": "equality",
                    "target_elements": ["side_AB"],
                    "value": 10,
                    "unit": "cm",
                    "text_source": "AB = 10 cm",
                    "confidence": 0.95,
                },
                {
                    "constraint_id": "c2",
                    "constraint_type": "equality",
                    "target_elements": ["side_AC"],
                    "value": 10,
                    "unit": "cm",
                    "text_source": "AC = 10 cm",
                    "confidence": 0.95,
                },
                {
                    "constraint_id": "c3",
                    "constraint_type": "angle",
                    "target_elements": ["angle_BAC"],
                    "value": 30,  # NOT 60° as visual might suggest
                    "unit": "degrees",
                    "text_source": "angle BAC = 30°",
                    "confidence": 0.9,
                },
            ]
        }

        mock_llm_response = LLMResponse(
            content=f"```json\n{llm_response_json}\n```",
            model_used="gpt-4o-mini",
            provider="openai",
            tokens_used=200,
            cost_estimate=0.015,
            latency_ms=600,
        )
        mock_llm_service.generate_non_stream.return_value = mock_llm_response
        mock_json_parser.parse_json_from_text.return_value = llm_response_json

        result = await extraction_service.extract_constraints_from_text(question_text, "q_text_law")

        # Verify text constraints are extracted, not visual assumptions
        angle_constraints = [
            c for c in result.constraints if c.constraint_type == GeometricConstraintType.ANGLE
        ]
        bac_angles = [c for c in angle_constraints if "BAC" in c.target_elements[0]]

        assert len(bac_angles) >= 1
        # Should extract 30° from text, NOT assume 60° from "equilateral" visual
        assert bac_angles[0].value == 30.0
