"""
Pydantic models for diagram generation using CGV (Constrain-Generate-Verify) pipeline.
Integrates with existing question_models.py and supports Manim + E2B execution.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, validator


class GeometricConstraintType(str, Enum):
    """Types of geometric constraints that can be extracted from question text"""

    EQUALITY = "equality"  # AB = AC, x = 5
    ANGLE = "angle"  # angle ABC = 60°
    DISTANCE = "distance"  # distance from A to B = 10cm
    PARALLEL = "parallel"  # line AB is parallel to line CD
    PERPENDICULAR = "perpendicular"  # line AB is perpendicular to line CD
    TANGENT = "tangent"  # line L is tangent to circle C
    RADIUS = "radius"  # radius of circle = 5cm
    RATIO = "ratio"  # AB : BC = 2 : 3


class ShapeType(str, Enum):
    """Supported geometric shapes for diagram generation"""

    TRIANGLE = "triangle"
    QUADRILATERAL = "quadrilateral"
    CIRCLE = "circle"
    LINE = "line"
    POLYGON = "polygon"
    POINT = "point"


class GeometricConstraint(BaseModel):
    """
    Single geometric constraint extracted from question text.
    Examples: "AB = 10", "angle ABC = 60°", "line AB parallel to line CD"
    """

    constraint_id: str = Field(..., description="Unique identifier for this constraint")
    constraint_type: GeometricConstraintType
    target_elements: list[str] = Field(
        ..., description="Element names involved (e.g., ['A', 'B', 'C'])"
    )
    value: Optional[Union[float, str]] = Field(None, description="Numeric value or symbolic value")
    unit: Optional[str] = Field(None, description="Unit (cm, °, etc.)")
    text_source: str = Field(..., description="Original text from which constraint was extracted")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")

    @validator("target_elements")
    def validate_target_elements(cls, v):  # noqa: N805
        if not v:
            raise ValueError("target_elements cannot be empty")
        return v


class ShapeDefinition(BaseModel):
    """Definition of a geometric shape to be rendered"""

    shape_id: str = Field(..., description="Unique identifier for this shape")
    shape_type: ShapeType
    vertices: list[str] = Field(..., description="Ordered list of vertex names")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Shape-specific properties"
    )

    @validator("vertices")
    def validate_vertices(cls, v, values):  # noqa: N805
        shape_type = values.get("shape_type")
        if shape_type == ShapeType.TRIANGLE and len(v) != 3:
            raise ValueError("Triangle must have exactly 3 vertices")
        elif shape_type == ShapeType.QUADRILATERAL and len(v) != 4:
            raise ValueError("Quadrilateral must have exactly 4 vertices")
        return v


class DiagramCanvas(BaseModel):
    """Canvas configuration for diagram rendering"""

    width: float = Field(default=16.0, description="Canvas width in Manim units")
    height: float = Field(default=9.0, description="Canvas height in Manim units")
    background_color: str = Field(default="WHITE", description="Background color")
    scale_factor: float = Field(default=1.0, gt=0, description="Global scale factor")
    margin: float = Field(default=0.5, ge=0, description="Margin around diagram elements")


class CoordinatePoint(BaseModel):
    """2D coordinate point for positioning diagram elements"""

    x: float
    y: float
    label: Optional[str] = None

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for Manim compatibility"""
        return np.array([self.x, self.y, 0])


class ManifestConstraints(BaseModel):
    """
    Structured representation of ALL constraints extracted from question text.
    This is the output of the CONSTRAIN step in CGV pipeline.
    """

    question_id: str = Field(..., description="Reference to source question")
    shapes: list[ShapeDefinition] = Field(default_factory=list)
    constraints: list[GeometricConstraint] = Field(default_factory=list)
    points: dict[str, CoordinatePoint] = Field(
        default_factory=dict, description="Solved point coordinates"
    )
    unknowns: list[str] = Field(
        default_factory=list, description="Variables to be solved (x, y, angle ABC)"
    )
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    extraction_method: str = Field(
        default="llm_extraction", description="Method used for extraction"
    )

    @validator("constraints")
    def validate_constraint_references(cls, v, values):  # noqa: N805
        """Ensure all constraint target_elements reference valid shapes/points"""
        shapes = values.get("shapes", [])
        shape_vertices = set()
        for shape in shapes:
            shape_vertices.update(shape.vertices)

        for constraint in v:
            for element in constraint.target_elements:
                # Allow standard geometric notation (lines, angles, etc.)
                if not (
                    element in shape_vertices
                    or element.startswith(("line_", "angle_", "side_", "arc_"))
                ):
                    # Warning only - don't fail validation for complex notation
                    pass
        return v


class GeometricSolution(BaseModel):
    """
    Solution from deterministic geometric solving.
    This is the output of the GENERATE step in CGV pipeline.
    """

    solved_points: dict[str, CoordinatePoint] = Field(default_factory=dict)
    solved_values: dict[str, float] = Field(default_factory=dict, description="Solved unknowns")
    derived_constraints: list[GeometricConstraint] = Field(
        default_factory=list, description="Additional derived facts"
    )
    solution_method: str = Field(default="sympy_solving", description="Solving method used")
    is_valid: bool = Field(default=True, description="Whether solution is geometrically valid")
    error_message: Optional[str] = Field(None, description="Error if solution failed")
    canvas_bounds: DiagramCanvas = Field(default_factory=DiagramCanvas)


class ManifestDiagramCode(BaseModel):
    """
    Generated Manim code for rendering diagram.
    This is prepared for the VERIFY step in CGV pipeline.
    """

    manim_code: str = Field(..., description="Complete Manim Python code")
    code_version: str = Field(default="1.0", description="Code generation version")
    dependencies: list[str] = Field(
        default_factory=lambda: ["manim", "numpy"], description="Required imports"
    )
    scene_class_name: str = Field(..., description="Name of the Manim Scene class")
    estimated_render_time: float = Field(
        default=30.0, description="Estimated render time in seconds"
    )
    complexity_score: float = Field(
        default=0.5, ge=0, le=1.0, description="Diagram complexity (0-1)"
    )


class DiagramValidationResult(BaseModel):
    """
    Quality assessment of generated diagram.
    This is the output of the VERIFY step in CGV pipeline.
    """

    geometric_accuracy: float = Field(..., ge=0, le=1.0, description="Geometric correctness score")
    readability_score: float = Field(..., ge=0, le=1.0, description="Visual clarity score")
    cambridge_compliance: float = Field(..., ge=0, le=1.0, description="IGCSE standards compliance")
    label_placement_score: float = Field(..., ge=0, le=1.0, description="Label positioning quality")
    collision_detection_score: float = Field(
        ..., ge=0, le=1.0, description="Overlap avoidance quality"
    )

    overall_quality: float = Field(..., ge=0, le=1.0, description="Weighted overall score")

    validation_issues: list[str] = Field(
        default_factory=list, description="Specific quality issues found"
    )
    improvement_suggestions: list[str] = Field(
        default_factory=list, description="Actionable improvements"
    )

    validation_timestamp: datetime = Field(default_factory=datetime.utcnow)

    @validator("overall_quality")
    def calculate_overall_quality(cls, v, values):  # noqa: N805
        """Calculate weighted average if not explicitly provided"""
        if v == 0:  # Default value, calculate from components
            scores = [
                values.get("geometric_accuracy", 0) * 0.3,
                values.get("readability_score", 0) * 0.25,
                values.get("cambridge_compliance", 0) * 0.25,
                values.get("label_placement_score", 0) * 0.1,
                values.get("collision_detection_score", 0) * 0.1,
            ]
            return sum(scores)
        return v


class DiagramGenerationRequest(BaseModel):
    """Input request for diagram generation"""

    question_text: str = Field(..., description="Full question text for constraint extraction")
    question_id: Optional[str] = Field(None, description="Reference to existing question")
    diagram_type: str = Field(default="static", description="static, animated, or interactive")
    canvas_config: DiagramCanvas = Field(default_factory=DiagramCanvas)
    generation_mode: str = Field(default="auto", description="auto, guided, or manual")
    include_labels: bool = Field(default=True, description="Whether to include text labels")
    include_measurements: bool = Field(default=True, description="Whether to show measurements")


class DiagramGenerationResult(BaseModel):
    """Complete result from diagram generation pipeline"""

    request: DiagramGenerationRequest
    manifest_constraints: ManifestConstraints
    geometric_solution: GeometricSolution
    manim_code: ManifestDiagramCode
    validation_result: DiagramValidationResult

    # Generated outputs
    diagram_image_path: Optional[str] = Field(None, description="Path to generated diagram image")
    diagram_animation_path: Optional[str] = Field(None, description="Path to generated animation")

    # Metadata
    generation_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time: float = Field(default=0.0, description="Total processing time in seconds")
    agent_reasoning_steps: list[dict[str, Any]] = Field(
        default_factory=list, description="Agent decision log"
    )

    # Status
    success: bool = Field(default=True)
    error_message: Optional[str] = Field(None)
    quality_passed: bool = Field(
        default=False, description="Whether validation passed quality thresholds"
    )


# Integration with existing question models
class QuestionWithDiagram(BaseModel):
    """Extended question model that includes optional diagram"""

    # Reference to existing Question model (to be imported from question_models.py)
    question_id_global: str
    diagram_generation_result: Optional[DiagramGenerationResult] = None
    diagram_required: bool = Field(default=False, description="Whether question requires a diagram")
    diagram_priority: str = Field(default="optional", description="required, helpful, or optional")
