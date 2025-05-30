"""
Pydantic models for question generation system.
Defines data structures for candidate questions, generation parameters, and database records.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum


class GenerationStatus(str, Enum):
    """Status of a generated candidate question"""
    CANDIDATE = "candidate"
    HUMAN_REVIEWED_ACCEPTED = "human_reviewed_accepted"
    HUMAN_REVIEWED_REJECTED = "human_reviewed_rejected"
    LLM_REVIEWED_NEEDS_HUMAN = "llm_reviewed_needs_human"
    AUTO_REJECTED = "auto_rejected"


class LLMModel(str, Enum):
    """Available LLM models"""
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet"  # Legacy - deprecated
    CLAUDE_4_SONNET = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    CLAUDE_4_OPUS = "us.anthropic.claude-opus-4-20250514-v1:0"
    GEMINI_PRO = "gemini-pro"
    GEMINI_FLASH = "gemini-flash"
    DEEPSEEK_R1 = "deepseek-ai/DeepSeek-R1-0528"
    QWEN3_235B = "Qwen/Qwen3-235B-A22B"


class CommandWord(str, Enum):
    """Available command words from Cambridge syllabus"""
    CALCULATE = "Calculate"
    CONSTRUCT = "Construct"
    DETERMINE = "Determine"
    DESCRIBE = "Describe"
    EXPLAIN = "Explain"
    GIVE = "Give"
    PLOT = "Plot"
    SHOW = "Show (that)"
    SKETCH = "Sketch"
    STATE = "State"
    WORK_OUT = "Work out"
    WRITE = "Write"
    WRITE_DOWN = "Write down"


class CalculatorPolicy(str, Enum):
    """Calculator usage policy"""
    ALLOWED = "allowed"
    NOT_ALLOWED = "not_allowed"
    VARIES_BY_QUESTION = "varies_by_question"


class GenerationConfig(BaseModel):
    """Configuration for a single question generation
    llm_model_generation, llm_model_marking_scheme, and llm_model_review can be any value from LLMModel, including Hugging Face models (deepseek-ai/DeepSeek-R1-0528, Qwen/Qwen3-235B-A22B).
    """
    generation_id: UUID = Field(default_factory=uuid4)
    seed_question_id: Optional[str] = None
    target_grade: int = Field(ge=1, le=9)
    calculator_policy: CalculatorPolicy
    desired_marks: int = Field(ge=1, le=5)
    subject_content_references: List[str] = Field(min_items=1)
    command_word_override: Optional[CommandWord] = None

    # Model configuration
    llm_model_generation: LLMModel = LLMModel.GPT_4O
    llm_model_marking_scheme: LLMModel = LLMModel.GPT_4O
    llm_model_review: LLMModel = LLMModel.CLAUDE_4_SONNET

    # Prompt versions
    prompt_template_version_generation: str = "v1.0"
    prompt_template_version_marking_scheme: str = "v1.0"
    prompt_template_version_review: str = "v1.0"

    # Generation parameters
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4000, ge=100, le=8000)


class AnswerSummary(BaseModel):
    """Final answer summary for a question"""
    answer_text: str
    value_numeric: Optional[float] = None
    unit: Optional[str] = None


class MarkAllocationCriterion(BaseModel):
    """Individual marking criterion"""
    criterion_id: str
    criterion_text: str
    mark_code_display: str
    marks_value: float
    mark_type_primary: Optional[Literal["M", "A", "B", "FT", "SC", "ISW", "OE", "SOI", "CAO", "DEP", "C", "E", "NA"]] = None
    qualifiers_and_notes: Optional[str] = None


class SolutionAndMarkingScheme(BaseModel):
    """Solution and marking scheme for a question"""
    final_answers_summary: List[AnswerSummary]
    mark_allocation_criteria: List[MarkAllocationCriterion]
    total_marks_for_part: int


class SolverStep(BaseModel):
    """Individual step in solver algorithm"""
    step_number: int
    description_text: str
    mathematical_expression_latex: Optional[str] = None
    skill_applied_tag: Optional[str] = None
    justification_or_reasoning: Optional[str] = None


class SolverAlgorithm(BaseModel):
    """Step-by-step solver algorithm"""
    steps: List[SolverStep]


class QuestionTaxonomy(BaseModel):
    """Taxonomic classification of a question"""
    topic_path: List[str] = Field(min_items=1)
    subject_content_references: List[str] = Field(min_items=1)
    skill_tags: List[str] = Field(min_items=1)
    cognitive_level: Optional[Literal["Recall", "ProceduralFluency", "ConceptualUnderstanding", "Application", "ProblemSolving", "Analysis"]] = None
    difficulty_estimate_0_to_1: Optional[float] = Field(None, ge=0.0, le=1.0)


class CandidateQuestion(BaseModel):
    """Generated candidate question following the schema"""
    # Core question data
    question_id_local: str
    question_id_global: str
    question_number_display: str
    marks: int
    command_word: CommandWord
    raw_text_content: str
    formatted_text_latex: Optional[str] = None

    # Classification and solutions
    taxonomy: QuestionTaxonomy
    solution_and_marking_scheme: SolutionAndMarkingScheme
    solver_algorithm: SolverAlgorithm

    # Generation metadata
    generation_id: UUID
    seed_question_id: Optional[str] = None
    target_grade_input: int

    # Model tracking
    llm_model_used_generation: str
    llm_model_used_marking_scheme: str
    llm_model_used_review: Optional[str] = None

    # Prompt tracking
    prompt_template_version_generation: str
    prompt_template_version_marking_scheme: str
    prompt_template_version_review: Optional[str] = None

    # Status and review
    generation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: GenerationStatus = GenerationStatus.CANDIDATE
    reviewer_notes: Optional[str] = None

    # Quality metrics
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    validation_errors: List[str] = Field(default_factory=list)


class GenerationRequest(BaseModel):
    """Request to generate candidate questions"""
    seed_question_id: Optional[str] = None
    target_grades: List[int] = Field(min_items=1)
    count_per_grade: int = Field(default=1, ge=1, le=10)
    subject_content_references: Optional[List[str]] = None
    calculator_policy: CalculatorPolicy = CalculatorPolicy.NOT_ALLOWED
    generation_config: Optional[GenerationConfig] = None


class GenerationResponse(BaseModel):
    """Response from question generation"""
    request_id: UUID = Field(default_factory=uuid4)
    generated_questions: List[CandidateQuestion]
    failed_generations: List[Dict[str, Any]] = Field(default_factory=list)
    total_requested: int
    total_generated: int
    generation_time_seconds: float
