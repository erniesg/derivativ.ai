"""
Strict enums for Cambridge IGCSE Mathematics question generation.
Based on official Cambridge syllabus (2025-2027) and actual past papers.
All syllabus references, command words, and taxonomies are strictly defined.
"""

from enum import Enum


class CommandWord(str, Enum):
    """Cambridge IGCSE Mathematics command words - STRICTLY from official syllabus"""
    CALCULATE = "Calculate"
    CONSTRUCT = "Construct"
    DETERMINE = "Determine"
    DESCRIBE = "Describe"
    EXPLAIN = "Explain"
    FIND = "Find"
    GIVE = "Give"
    PLOT = "Plot"
    SHOW_THAT = "Show (that)"
    SKETCH = "Sketch"
    STATE = "State"
    WORK_OUT = "Work out"
    WRITE = "Write"
    WRITE_DOWN = "Write down"


class SubjectContentReference(str, Enum):
    """Cambridge IGCSE Mathematics subject content references - STRICTLY from 2025-2027 syllabus"""
    # Core Number (C1)
    C1_1 = "C1.1"   # Types of number
    C1_2 = "C1.2"   # Sets
    C1_3 = "C1.3"   # Powers and roots
    C1_4 = "C1.4"   # Fractions, decimals and percentages
    C1_5 = "C1.5"   # Ordering
    C1_6 = "C1.6"   # The four operations
    C1_7 = "C1.7"   # Indices I
    C1_8 = "C1.8"   # Standard form
    C1_9 = "C1.9"   # Estimation
    C1_10 = "C1.10" # Limits of accuracy
    C1_11 = "C1.11" # Ratio and proportion
    C1_12 = "C1.12" # Rates
    C1_13 = "C1.13" # Percentages
    C1_14 = "C1.14" # Using a calculator
    C1_15 = "C1.15" # Time
    C1_16 = "C1.16" # Money
    
    # Core Algebra and graphs (C2)
    C2_1 = "C2.1"   # Introduction to algebra
    C2_2 = "C2.2"   # Algebraic manipulation
    C2_4 = "C2.4"   # Indices II
    C2_5 = "C2.5"   # Equations
    C2_6 = "C2.6"   # Inequalities
    C2_7 = "C2.7"   # Sequences
    C2_9 = "C2.9"   # Graphs in practical situations
    C2_10 = "C2.10" # Graphs of functions
    C2_11 = "C2.11" # Sketching curves
    
    # Core Coordinate geometry (C3)
    C3_1 = "C3.1"   # Coordinates
    C3_2 = "C3.2"   # Drawing linear graphs
    C3_3 = "C3.3"   # Gradient of linear graphs
    C3_5 = "C3.5"   # Equations of linear graphs
    C3_6 = "C3.6"   # Parallel lines
    
    # Core Geometry (C4)
    C4_1 = "C4.1"   # Geometrical terms
    C4_2 = "C4.2"   # Geometrical constructions
    C4_3 = "C4.3"   # Scale drawings
    C4_4 = "C4.4"   # Similarity
    C4_5 = "C4.5"   # Symmetry
    C4_6 = "C4.6"   # Angles
    C4_7 = "C4.7"   # Circle theorems
    
    # Core Mensuration (C5)
    C5_1 = "C5.1"   # Units of measure
    C5_2 = "C5.2"   # Area and perimeter
    C5_3 = "C5.3"   # Circles, arcs and sectors
    C5_4 = "C5.4"   # Surface area and volume
    C5_5 = "C5.5"   # Compound shapes and parts of shapes
    
    # Core Trigonometry (C6)
    C6_1 = "C6.1"   # Pythagoras' theorem
    C6_2 = "C6.2"   # Right-angled triangles
    
    # Core Transformations and vectors (C7)
    C7_1 = "C7.1"   # Transformations
    
    # Core Probability (C8)
    C8_1 = "C8.1"   # Introduction to probability
    C8_2 = "C8.2"   # Relative and expected frequencies
    C8_3 = "C8.3"   # Probability of combined events
    
    # Core Statistics (C9)
    C9_1 = "C9.1"   # Classifying statistical data
    C9_2 = "C9.2"   # Interpreting statistical data
    C9_3 = "C9.3"   # Averages and range
    C9_4 = "C9.4"   # Statistical charts and diagrams
    C9_5 = "C9.5"   # Scatter diagrams
    
    # Extended Number (E1)
    E1_1 = "E1.1"   # Types of number
    E1_2 = "E1.2"   # Sets
    E1_3 = "E1.3"   # Powers and roots
    E1_4 = "E1.4"   # Fractions, decimals and percentages
    E1_5 = "E1.5"   # Ordering
    E1_6 = "E1.6"   # The four operations
    E1_7 = "E1.7"   # Indices I
    E1_8 = "E1.8"   # Standard form
    E1_9 = "E1.9"   # Estimation
    E1_10 = "E1.10" # Limits of accuracy
    E1_11 = "E1.11" # Ratio and proportion
    E1_12 = "E1.12" # Rates
    E1_13 = "E1.13" # Percentages
    E1_14 = "E1.14" # Using a calculator
    E1_15 = "E1.15" # Time
    E1_16 = "E1.16" # Money
    E1_17 = "E1.17" # Exponential growth and decay
    E1_18 = "E1.18" # Surds
    
    # Extended Algebra and graphs (E2)
    E2_1 = "E2.1"   # Introduction to algebra
    E2_2 = "E2.2"   # Algebraic manipulation
    E2_3 = "E2.3"   # Algebraic fractions
    E2_4 = "E2.4"   # Indices II
    E2_5 = "E2.5"   # Equations
    E2_6 = "E2.6"   # Inequalities
    E2_7 = "E2.7"   # Sequences
    E2_8 = "E2.8"   # Proportion
    E2_9 = "E2.9"   # Graphs in practical situations
    E2_10 = "E2.10" # Graphs of functions
    E2_11 = "E2.11" # Sketching curves
    E2_12 = "E2.12" # Differentiation
    E2_13 = "E2.13" # Functions
    
    # Extended Coordinate geometry (E3)
    E3_1 = "E3.1"   # Coordinates
    E3_2 = "E3.2"   # Drawing linear graphs
    E3_3 = "E3.3"   # Gradient of linear graphs
    E3_4 = "E3.4"   # Length and midpoint
    E3_5 = "E3.5"   # Equations of linear graphs
    E3_6 = "E3.6"   # Parallel lines
    E3_7 = "E3.7"   # Perpendicular lines
    
    # Extended Geometry (E4)
    E4_1 = "E4.1"   # Geometrical terms
    E4_2 = "E4.2"   # Geometrical constructions
    E4_3 = "E4.3"   # Scale drawings
    E4_4 = "E4.4"   # Similarity
    E4_5 = "E4.5"   # Symmetry
    E4_6 = "E4.6"   # Angles
    E4_7 = "E4.7"   # Circle theorems I
    E4_8 = "E4.8"   # Circle theorems II
    
    # Extended Mensuration (E5)
    E5_1 = "E5.1"   # Units of measure
    E5_2 = "E5.2"   # Area and perimeter
    E5_3 = "E5.3"   # Circles, arcs and sectors
    E5_4 = "E5.4"   # Surface area and volume
    E5_5 = "E5.5"   # Compound shapes and parts of shapes
    
    # Extended Trigonometry (E6)
    E6_1 = "E6.1"   # Pythagoras' theorem
    E6_2 = "E6.2"   # Right-angled triangles
    E6_3 = "E6.3"   # Exact trigonometric values
    E6_4 = "E6.4"   # Trigonometric functions
    E6_5 = "E6.5"   # Non-right-angled triangles
    E6_6 = "E6.6"   # Pythagoras' theorem and trigonometry in 3D
    
    # Extended Transformations and vectors (E7)
    E7_1 = "E7.1"   # Transformations
    E7_2 = "E7.2"   # Vectors in two dimensions
    E7_3 = "E7.3"   # Magnitude of a vector
    E7_4 = "E7.4"   # Vector geometry
    
    # Extended Probability (E8)
    E8_1 = "E8.1"   # Introduction to probability
    E8_2 = "E8.2"   # Relative and expected frequencies
    E8_3 = "E8.3"   # Probability of combined events
    E8_4 = "E8.4"   # Conditional probability
    
    # Extended Statistics (E9)
    E9_1 = "E9.1"   # Classifying statistical data
    E9_2 = "E9.2"   # Interpreting statistical data
    E9_3 = "E9.3"   # Averages and measures of spread
    E9_4 = "E9.4"   # Statistical charts and diagrams
    E9_5 = "E9.5"   # Scatter diagrams
    E9_6 = "E9.6"   # Cumulative frequency diagrams
    E9_7 = "E9.7"   # Histograms


class TopicName(str, Enum):
    """Cambridge IGCSE Mathematics topic names - STRICTLY from syllabus"""
    NUMBER = "Number"
    ALGEBRA_AND_GRAPHS = "Algebra and graphs"
    COORDINATE_GEOMETRY = "Coordinate geometry"
    GEOMETRY = "Geometry"
    MENSURATION = "Mensuration"
    TRIGONOMETRY = "Trigonometry"
    TRANSFORMATIONS_AND_VECTORS = "Transformations and vectors"
    PROBABILITY = "Probability"
    STATISTICS = "Statistics"


class CalculatorPolicy(str, Enum):
    """Calculator usage policy for questions"""
    ALLOWED = "allowed"
    NOT_ALLOWED = "not_allowed"
    VARIES_BY_QUESTION = "varies_by_question"


class Tier(str, Enum):
    """Cambridge IGCSE Mathematics paper tiers"""
    CORE = "Core"
    EXTENDED = "Extended"


class CognitiveLevel(str, Enum):
    """Cognitive complexity levels for questions"""
    RECALL = "Recall"
    PROCEDURAL_FLUENCY = "ProceduralFluency"
    CONCEPTUAL_UNDERSTANDING = "ConceptualUnderstanding"
    APPLICATION = "Application"
    PROBLEM_SOLVING = "ProblemSolving"
    ANALYSIS = "Analysis"


class QuestionOrigin(str, Enum):
    """Source of the question"""
    PAST_PAPER = "past_paper"
    GENERATED = "generated"


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENROUTER = "openrouter"


class LLMModel(str, Enum):
    """Specific LLM models for question generation"""
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"
    GEMINI_PRO = "gemini-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"


class QualityAction(str, Enum):
    """Actions from quality control workflow"""
    APPROVE = "approve"
    MANUAL_REVIEW = "manual_review"
    REFINE = "refine"
    REGENERATE = "regenerate"
    REJECT = "reject"


class DiagramType(str, Enum):
    """Types of diagrams that can be generated with Manim"""
    GEOMETRIC_ANGLE_PROBLEM = "geometric_angle_problem"
    NET_OF_SOLID = "net_of_solid"
    COORDINATE_GEOMETRY_PLOT = "coordinate_geometry_plot"
    SCATTER_PLOT = "scatter_plot"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    LINE_GRAPH = "line_graph"
    HISTOGRAM = "histogram"
    VENN_DIAGRAM = "venn_diagram"
    TREE_DIAGRAM = "tree_diagram"
    NUMBER_LINE = "number_line"
    SCALE_DRAWING = "scale_drawing"
    TRANSFORMATION_GRID = "transformation_grid"
    GENERIC_ILLUSTRATION = "generic_illustration"


# Validation and utility functions
def validate_subject_content_ref(ref: str) -> bool:
    """Validate if a subject content reference is valid"""
    try:
        SubjectContentReference(ref)
        return True
    except ValueError:
        return False


def get_core_subject_refs() -> list[str]:
    """Get all core subject content references"""
    return [ref.value for ref in SubjectContentReference if ref.value.startswith("C")]


def get_extended_subject_refs() -> list[str]:
    """Get all extended subject content references"""
    return [ref.value for ref in SubjectContentReference if ref.value.startswith("E")]


def get_command_words() -> list[str]:
    """Get all valid command words"""
    return [word.value for word in CommandWord]


def get_topic_names() -> list[str]:
    """Get all valid topic names"""
    return [topic.value for topic in TopicName]


def get_tier_refs_for_tier(tier: Tier) -> list[str]:
    """Get appropriate subject content references for a tier"""
    if tier == Tier.CORE:
        return get_core_subject_refs()
    elif tier == Tier.EXTENDED:
        return get_core_subject_refs() + get_extended_subject_refs()
    else:
        return get_core_subject_refs()


def refs_by_topic(topic: TopicName, tier: Tier = Tier.CORE) -> list[str]:
    """Get subject content references filtered by topic and tier"""
    topic_mapping = {
        TopicName.NUMBER: ["1"],
        TopicName.ALGEBRA_AND_GRAPHS: ["2"],
        TopicName.COORDINATE_GEOMETRY: ["3"],
        TopicName.GEOMETRY: ["4"],
        TopicName.MENSURATION: ["5"],
        TopicName.TRIGONOMETRY: ["6"],
        TopicName.TRANSFORMATIONS_AND_VECTORS: ["7"],
        TopicName.PROBABILITY: ["8"],
        TopicName.STATISTICS: ["9"],
    }
    
    tier_prefix = "C" if tier == Tier.CORE else "E"
    topic_numbers = topic_mapping.get(topic, [])
    
    all_refs = get_tier_refs_for_tier(tier)
    filtered_refs = []
    
    for ref in all_refs:
        for topic_num in topic_numbers:
            if ref.startswith(f"{tier_prefix}{topic_num}."):
                filtered_refs.append(ref)
                break
    
    return filtered_refs