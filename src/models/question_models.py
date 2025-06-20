"""
Pydantic models for Cambridge IGCSE Mathematics question generation.
Designed for smolagents + Modal architecture with strict Cambridge compliance.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .enums import (
    CalculatorPolicy,
    CognitiveLevel,
    CommandWord,
    LLMModel,
    QualityAction,
    QuestionOrigin,
    SubjectContentReference,
    Tier,
)


class GenerationStatus(str, Enum):
    """Status of a generated candidate question"""

    CANDIDATE = "candidate"
    HUMAN_REVIEWED_ACCEPTED = "human_reviewed_accepted"
    HUMAN_REVIEWED_REJECTED = "human_reviewed_rejected"
    LLM_REVIEWED_NEEDS_HUMAN = "llm_reviewed_needs_human"
    AUTO_REJECTED = "auto_rejected"


class MarkType(str, Enum):
    """Cambridge mark scheme mark types"""

    M = "M"  # Method mark
    A = "A"  # Accuracy mark
    B = "B"  # Independent/Bonus mark
    FT = "FT"  # Follow through mark
    SC = "SC"  # Special case mark
    ISW = "ISW"  # Ignore subsequent working
    OE = "OE"  # Or equivalent
    SOI = "SOI"  # Seen or implied
    CAO = "CAO"  # Correct answer only
    DEP = "DEP"  # Dependent mark
    C = "C"  # Communication mark
    E = "E"  # Explanation mark


# Core Question Models
class FinalAnswer(BaseModel):
    """Single final answer for a question part"""

    answer_text: str = Field(..., description="The answer as shown in mark scheme")
    value_numeric: Optional[float] = Field(None, description="Numeric value if applicable")
    unit: Optional[str] = Field(None, description="Unit if applicable")


class MarkingCriterion(BaseModel):
    """Single marking criterion from Cambridge mark scheme"""

    criterion_id: str = Field(..., description="Unique ID for this criterion")
    criterion_text: str = Field(..., description="What earns the marks")
    mark_code_display: str = Field(..., description="Mark code as displayed (e.g., 'M1', 'A1')")
    marks_value: int = Field(..., description="Number of marks for this criterion")
    mark_type_primary: Optional[MarkType] = Field(None, description="Primary mark type")
    qualifiers_and_notes: Optional[str] = Field(None, description="Additional conditions")


class SolutionAndMarkingScheme(BaseModel):
    """Complete solution and marking scheme for a question"""

    final_answers_summary: list[FinalAnswer] = Field(..., description="All final answers")
    mark_allocation_criteria: list[MarkingCriterion] = Field(..., description="Detailed marking")
    total_marks_for_part: int = Field(..., description="Total marks for this question part")


class SolverStep(BaseModel):
    """Single step in question solving algorithm"""

    step_number: int
    description_text: str = Field(..., description="What to do in this step")
    mathematical_expression_latex: Optional[str] = Field(
        None, description="Math formula/calculation"
    )
    skill_applied_tag: Optional[str] = Field(None, description="Skill used in this step")
    justification_or_reasoning: Optional[str] = Field(None, description="Why this step works")


class SolverAlgorithm(BaseModel):
    """Step-by-step solution algorithm"""

    steps: list[SolverStep] = Field(..., description="All solution steps")


class QuestionTaxonomy(BaseModel):
    """Cambridge IGCSE taxonomy for question classification"""

    topic_path: list[str] = Field(..., description="Topic hierarchy")
    subject_content_references: list[SubjectContentReference] = Field(
        ..., description="Syllabus codes"
    )
    skill_tags: list[str] = Field(..., description="Granular skills applied")
    cognitive_level: Optional[CognitiveLevel] = Field(None, description="Cognitive complexity")
    difficulty_estimate_0_to_1: Optional[float] = Field(
        None, ge=0, le=1, description="Difficulty rating"
    )


class AssetRecreationData(BaseModel):
    """Data for recreating diagrams with Manim"""

    diagram_type: Optional[str] = Field(None, description="Type of diagram")
    elements: list[dict[str, Any]] = Field(default_factory=list, description="Diagram elements")
    relationships: list[dict[str, Any]] = Field(
        default_factory=list, description="Element relationships"
    )
    layout_params: Optional[dict[str, Any]] = Field(None, description="Layout parameters")
    manim_script_path: Optional[str] = Field(None, description="Path to Manim script")
    manim_scene_class: Optional[str] = Field(None, description="Manim scene class name")


class QuestionAsset(BaseModel):
    """Asset (diagram, table, etc.) associated with a question"""

    asset_id_local: str = Field(..., description="Local asset identifier")
    asset_type: str = Field(..., description="Type of asset")
    description_for_accessibility: str = Field(..., description="Alt text description")
    recreation_data: Optional[AssetRecreationData] = Field(
        None, description="Manim recreation data"
    )


class Question(BaseModel):
    """Complete Cambridge IGCSE Mathematics question"""

    question_id_local: str = Field(..., description="Question ID within paper")
    question_id_global: str = Field(..., description="Globally unique question ID")
    question_number_display: str = Field(..., description="Display number (e.g., '1 (a)')")
    marks: int = Field(..., ge=1, le=20, description="Total marks for this question")
    command_word: CommandWord = Field(..., description="Cambridge command word")
    raw_text_content: str = Field(..., min_length=10, description="Question text")
    formatted_text_latex: Optional[str] = Field(None, description="LaTeX formatted text")
    assets: list[QuestionAsset] = Field(default_factory=list, description="Associated assets")
    taxonomy: QuestionTaxonomy = Field(..., description="Question classification")
    solution_and_marking_scheme: SolutionAndMarkingScheme = Field(..., description="Solution")
    solver_algorithm: SolverAlgorithm = Field(..., description="Step-by-step solution")


# Generation Request Models
class GenerationRequest(BaseModel):
    """Request for generating new questions"""

    topic: str = Field(..., min_length=2, description="Topic to generate questions about")
    tier: Tier = Field(default=Tier.CORE, description="Core or Extended")
    grade_level: Optional[int] = Field(None, ge=1, le=9, description="Target grade")
    marks: int = Field(default=3, ge=1, le=20, description="Target marks")
    count: int = Field(default=1, ge=1, le=10, description="Number of questions")
    calculator_policy: CalculatorPolicy = Field(default=CalculatorPolicy.NOT_ALLOWED)
    subject_content_refs: Optional[list[SubjectContentReference]] = Field(
        None, description="Specific syllabus refs"
    )
    command_word: Optional[CommandWord] = Field(None, description="Specific command word")
    cognitive_level: Optional[CognitiveLevel] = Field(None, description="Target cognitive level")
    include_diagrams: bool = Field(default=False, description="Whether to include diagrams")

    # Generation settings
    llm_model: LLMModel = Field(default=LLMModel.GPT_4O, description="LLM model to use")
    temperature: float = Field(default=0.7, ge=0, le=2, description="Generation temperature")
    max_retries: int = Field(default=3, ge=1, le=10, description="Max generation retries")


class AgentResult(BaseModel):
    """Result from a single agent operation"""

    success: bool = Field(..., description="Whether operation succeeded")
    agent_name: str = Field(..., description="Name of the agent")
    output: Optional[dict[str, Any]] = Field(None, description="Agent output data")
    error: Optional[str] = Field(None, description="Error message if failed")
    reasoning_steps: list[str] = Field(default_factory=list, description="Agent reasoning")
    processing_time: float = Field(default=0.0, description="Time taken in seconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class QualityDecision(BaseModel):
    """Decision from quality control workflow"""

    action: QualityAction = Field(..., description="Recommended action")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in decision")
    reasoning: str = Field(..., description="Explanation for decision")
    quality_score: float = Field(..., ge=0, le=1, description="Overall quality score")
    suggested_improvements: list[str] = Field(
        default_factory=list, description="Improvement suggestions"
    )
    mathematical_accuracy: Optional[float] = Field(
        None, ge=0, le=1, description="Math accuracy score"
    )
    cambridge_compliance: Optional[float] = Field(
        None, ge=0, le=1, description="Syllabus compliance"
    )
    grade_appropriateness: Optional[float] = Field(
        None, ge=0, le=1, description="Grade level match"
    )


class GenerationSession(BaseModel):
    """Complete question generation session"""

    session_id: UUID = Field(default_factory=uuid4, description="Unique session ID")
    request: GenerationRequest = Field(..., description="Original generation request")
    questions: list[Question] = Field(default_factory=list, description="Generated questions")
    quality_decisions: list[QualityDecision] = Field(
        default_factory=list, description="Quality assessments"
    )
    agent_results: list[AgentResult] = Field(default_factory=list, description="All agent results")
    status: GenerationStatus = Field(
        default=GenerationStatus.CANDIDATE, description="Session status"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    total_processing_time: float = Field(default=0.0, description="Total time taken")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Session metadata")


# Database Models
class QuestionDB(BaseModel):
    """Database record for a question"""

    id: Optional[int] = Field(None, description="Database primary key")
    question_id_global: str = Field(..., description="Global question identifier")
    content_json: dict[str, Any] = Field(..., description="Full question data as JSON")
    tier: Tier = Field(..., description="Core or Extended")
    marks: int = Field(..., description="Total marks")
    command_word: CommandWord = Field(..., description="Command word used")
    subject_content_refs: list[str] = Field(..., description="Syllabus references")
    cognitive_level: Optional[CognitiveLevel] = Field(None, description="Cognitive level")
    difficulty_estimate: Optional[float] = Field(None, description="Difficulty rating")
    quality_score: Optional[float] = Field(None, description="Quality assessment")
    origin: QuestionOrigin = Field(..., description="Source of question")
    status: GenerationStatus = Field(..., description="Current status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# Validation helpers
def validate_question_structure(question: Question) -> bool:
    """Validate that question follows Cambridge IGCSE structure"""
    # Check marks allocation matches marking scheme
    scheme_total = question.solution_and_marking_scheme.total_marks_for_part
    if question.marks != scheme_total:
        return False

    # Check marking criteria add up
    criteria_total = sum(
        c.marks_value for c in question.solution_and_marking_scheme.mark_allocation_criteria
    )
    if criteria_total != scheme_total:
        return False

    # Check syllabus references are valid
    for ref in question.taxonomy.subject_content_references:
        try:
            SubjectContentReference(ref.value)
        except ValueError:
            return False

    return True
