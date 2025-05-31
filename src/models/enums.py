"""
Enums for Cambridge IGCSE Mathematics question generation.
Based on official Cambridge syllabus (data/syllabus_command.json) and actual past papers.
"""

from enum import Enum


class SkillTag(Enum):
    """Valid skill tags extracted from Cambridge IGCSE Mathematics past papers"""
    ADDITION = "ADDITION"
    ALTERNATE_ANGLES = "ALTERNATE_ANGLES"
    ANGLES_IN_A_TRIANGLE = "ANGLES_IN_A_TRIANGLE"
    ANGLES_ON_A_STRAIGHT_LINE = "ANGLES_ON_A_STRAIGHT_LINE"
    ANGLE_OF_ROTATION = "ANGLE_OF_ROTATION"
    ANGLE_PROPERTIES = "ANGLE_PROPERTIES"
    AREA_COMPOSITE_SHAPES = "AREA_COMPOSITE_SHAPES"
    AREA_PARALLELOGRAM = "AREA_PARALLELOGRAM"
    BEARINGS = "BEARINGS"
    CENTER_OF_ROTATION = "CENTER_OF_ROTATION"
    CENTRE_OF_ENLARGEMENT = "CENTRE_OF_ENLARGEMENT"
    CONSTRUCTION = "CONSTRUCTION"
    CORRELATION_TYPE = "CORRELATION_TYPE"
    DESCRIBE_TRANSFORMATION = "DESCRIBE_TRANSFORMATION"
    DIRECTION_OF_ROTATION = "DIRECTION_OF_ROTATION"
    DURATION = "DURATION"
    ELIMINATION_METHOD = "ELIMINATION_METHOD"
    ENLARGEMENT = "ENLARGEMENT"
    EQUATION_OF_A_LINE = "EQUATION_OF_A_LINE"
    ESTIMATION_USING_LINE_OF_BEST_FIT = "ESTIMATION_USING_LINE_OF_BEST_FIT"
    EVEN_NUMBERS = "EVEN_NUMBERS"
    FACTORISATION_COMMON_FACTOR = "FACTORISATION_COMMON_FACTOR"
    FORMING_NUMBERS = "FORMING_NUMBERS"
    FORM_EQUATION = "FORM_EQUATION"
    FRACTION_OF_QUANTITY = "FRACTION_OF_QUANTITY"
    GRADIENT = "GRADIENT"
    HCF = "HCF"
    IDENTIFY_SOLID_FROM_NET = "IDENTIFY_SOLID_FROM_NET"
    INEQUALITIES = "INEQUALITIES"
    INEQUALITY_FROM_NUMBER_LINE = "INEQUALITY_FROM_NUMBER_LINE"
    INTEGER_SOLUTIONS = "INTEGER_SOLUTIONS"
    INTERSECTION_OF_GRAPHS = "INTERSECTION_OF_GRAPHS"
    LIMITS_OF_ACCURACY = "LIMITS_OF_ACCURACY"
    LINE_OF_BEST_FIT_DRAWING = "LINE_OF_BEST_FIT_DRAWING"
    LINE_OF_SYMMETRY_QUADRATIC = "LINE_OF_SYMMETRY_QUADRATIC"
    LOGICAL_REASONING = "LOGICAL_REASONING"
    LOWER_BOUND = "LOWER_BOUND"
    LOWEST_COMMON_MULTIPLE_CONCEPT = "LOWEST_COMMON_MULTIPLE_CONCEPT"
    MEASUREMENT_ANGLE = "MEASUREMENT_ANGLE"
    MEASUREMENT_LENGTH = "MEASUREMENT_LENGTH"
    MEDIAN = "MEDIAN"
    MODE = "MODE"
    MULTIPLE = "MULTIPLE"
    MULTIPLICATION = "MULTIPLICATION"
    NEAREST_HUNDRED = "NEAREST_HUNDRED"
    NEGATIVE_INDICES = "NEGATIVE_INDICES"
    NUMBER_RANGE = "NUMBER_RANGE"
    NUMBER_TO_WORDS = "NUMBER_TO_WORDS"
    ORDERING_DATA = "ORDERING_DATA"
    ORDER_OF_OPERATIONS = "ORDER_OF_OPERATIONS"
    ORDER_OF_ROTATIONAL_SYMMETRY = "ORDER_OF_ROTATIONAL_SYMMETRY"
    PERCENTAGES_SUM_TO_100 = "PERCENTAGES_SUM_TO_100"
    PLACE_VALUE = "PLACE_VALUE"
    PLOTTING_GRAPH = "PLOTTING_GRAPH"
    PLOTTING_POINTS = "PLOTTING_POINTS"
    PRIME_FACTORIZATION = "PRIME_FACTORIZATION"
    PROBABILITY_AND_RULE = "PROBABILITY_AND_RULE"
    PROBABILITY_COMPLEMENT = "PROBABILITY_COMPLEMENT"
    QUADRATIC_FUNCTION = "QUADRATIC_FUNCTION"
    QUADRATIC_GRAPH = "QUADRATIC_GRAPH"
    RATIO_SIMPLIFICATION = "RATIO_SIMPLIFICATION"
    READING_FROM_GRAPH = "READING_FROM_GRAPH"
    REARRANGE_FORMULA = "REARRANGE_FORMULA"
    ROTATION = "ROTATION"
    ROTATIONAL_SYMMETRY = "ROTATIONAL_SYMMETRY"
    ROUNDING = "ROUNDING"
    SCALE_DRAWING = "SCALE_DRAWING"
    SCALE_FACTOR = "SCALE_FACTOR"
    SCATTER_DIAGRAM = "SCATTER_DIAGRAM"
    SCATTER_DIAGRAM_INTERPRETATION = "SCATTER_DIAGRAM_INTERPRETATION"
    SET_COMPLEMENT_CARDINALITY = "SET_COMPLEMENT_CARDINALITY"
    SET_INTERSECTION = "SET_INTERSECTION"
    SET_OPERATIONS = "SET_OPERATIONS"
    SIMULTANEOUS_LINEAR_EQUATIONS = "SIMULTANEOUS_LINEAR_EQUATIONS"
    SOLVE_LINEAR_EQUATION = "SOLVE_LINEAR_EQUATION"
    SQUARE_BASED_PYRAMID = "SQUARE_BASED_PYRAMID"
    SQUARE_NUMBER = "SQUARE_NUMBER"
    SUBSTITUTION = "SUBSTITUTION"
    SUBSTITUTION_METHOD = "SUBSTITUTION_METHOD"
    SUBTRACTION = "SUBTRACTION"
    SUBTRACTION_DECIMALS = "SUBTRACTION_DECIMALS"
    SUBTRACTION_NEGATIVE_NUMBERS = "SUBTRACTION_NEGATIVE_NUMBERS"
    SUBTRACTION_WHOLE_NUMBERS = "SUBTRACTION_WHOLE_NUMBERS"
    TABLE_OF_VALUES = "TABLE_OF_VALUES"
    TIME_CALCULATION = "TIME_CALCULATION"
    TIME_ZONES = "TIME_ZONES"
    TRANSLATION = "TRANSLATION"
    TREE_DIAGRAM_COMPLETION = "TREE_DIAGRAM_COMPLETION"
    TREE_DIAGRAM_USE = "TREE_DIAGRAM_USE"
    UNGROUPED_DATA = "UNGROUPED_DATA"
    UPPER_BOUND = "UPPER_BOUND"
    VECTOR_NOTATION = "VECTOR_NOTATION"
    VENN_DIAGRAM_COMPLETION = "VENN_DIAGRAM_COMPLETION"
    VERTICES_OF_SOLID = "VERTICES_OF_SOLID"
    VISUALIZATION = "VISUALIZATION"
    WORD_PROBLEM = "WORD_PROBLEM"
    Y_INTERCEPT = "Y_INTERCEPT"


class SubjectContentReference(Enum):
    """Valid Cambridge IGCSE Mathematics subject content references from official syllabus"""
    C1_1 = "C1.1"  # Types of number (Number - core)
    C1_10 = "C1.10"  # Limits of accuracy (Number - core)
    C1_11 = "C1.11"  # Ratio and proportion (Number - core)
    C1_12 = "C1.12"  # Rates (Number - core)
    C1_13 = "C1.13"  # Percentages (Number - core)
    C1_14 = "C1.14"  # Using a calculator (Number - core)
    C1_15 = "C1.15"  # Time (Number - core)
    C1_16 = "C1.16"  # Money (Number - core)
    C1_2 = "C1.2"  # Sets (Number - core)
    C1_3 = "C1.3"  # Powers and roots (Number - core)
    C1_4 = "C1.4"  # Fractions, decimals and percentages (Number - core)
    C1_5 = "C1.5"  # Ordering (Number - core)
    C1_6 = "C1.6"  # The four operations (Number - core)
    C1_7 = "C1.7"  # Indices I (Number - core)
    C1_8 = "C1.8"  # Standard form (Number - core)
    C1_9 = "C1.9"  # Estimation (Number - core)
    C2_1 = "C2.1"  # Introduction to algebra (Algebra and graphs - core)
    C2_10 = "C2.10"  # Graphs of functions (Algebra and graphs - core)
    C2_11 = "C2.11"  # Sketching curves (Algebra and graphs - core)
    C2_2 = "C2.2"  # Algebraic manipulation (Algebra and graphs - core)
    C2_4 = "C2.4"  # Indices II (Algebra and graphs - core)
    C2_5 = "C2.5"  # Equations (Algebra and graphs - core)
    C2_6 = "C2.6"  # Inequalities (Algebra and graphs - core)
    C2_7 = "C2.7"  # Sequences (Algebra and graphs - core)
    C2_9 = "C2.9"  # Graphs in practical situations (Algebra and graphs - core)
    C3_1 = "C3.1"  # Coordinates (Coordinate geometry - core)
    C3_2 = "C3.2"  # Drawing linear graphs (Coordinate geometry - core)
    C3_3 = "C3.3"  # Gradient of linear graphs (Coordinate geometry - core)
    C3_5 = "C3.5"  # Equations of linear graphs (Coordinate geometry - core)
    C3_6 = "C3.6"  # Parallel lines (Coordinate geometry - core)
    C4_1 = "C4.1"  # Geometrical terms (Geometry - core)
    C4_2 = "C4.2"  # Geometrical constructions (Geometry - core)
    C4_3 = "C4.3"  # Scale drawings (Geometry - core)
    C4_4 = "C4.4"  # Similarity (Geometry - core)
    C4_5 = "C4.5"  # Symmetry (Geometry - core)
    C4_6 = "C4.6"  # Angles (Geometry - core)
    C4_7 = "C4.7"  # Circle theorems (Geometry - core)
    C5_1 = "C5.1"  # Units of measure (Mensuration - core)
    C5_2 = "C5.2"  # Area and perimeter (Mensuration - core)
    C5_3 = "C5.3"  # Circles, arcs and sectors (Mensuration - core)
    C5_4 = "C5.4"  # Surface area and volume (Mensuration - core)
    C5_5 = "C5.5"  # Compound shapes and parts of shapes (Mensuration - core)
    C6_1 = "C6.1"  # Pythagoras' theorem (Trigonometry - core)
    C6_2 = "C6.2"  # Right-angled triangles (Trigonometry - core)
    C7_1 = "C7.1"  # Transformations (Transformations and vectors - core)
    C8_1 = "C8.1"  # Introduction to probability (Probability - core)
    C8_2 = "C8.2"  # Relative and expected frequencies (Probability - core)
    C8_3 = "C8.3"  # Probability of combined events (Probability - core)
    C9_1 = "C9.1"  # Classifying statistical data (Statistics - core)
    C9_2 = "C9.2"  # Interpreting statistical data (Statistics - core)
    C9_3 = "C9.3"  # Averages and range (Statistics - core)
    C9_4 = "C9.4"  # Statistical charts and diagrams (Statistics - core)
    C9_5 = "C9.5"  # Scatter diagrams (Statistics - core)
    E1_1 = "E1.1"  # Types of number (Number - extended)
    E1_10 = "E1.10"  # Limits of accuracy (Number - extended)
    E1_11 = "E1.11"  # Ratio and proportion (Number - extended)
    E1_12 = "E1.12"  # Rates (Number - extended)
    E1_13 = "E1.13"  # Percentages (Number - extended)
    E1_14 = "E1.14"  # Using a calculator (Number - extended)
    E1_15 = "E1.15"  # Time (Number - extended)
    E1_16 = "E1.16"  # Money (Number - extended)
    E1_17 = "E1.17"  # Exponential growth and decay (Number - extended)
    E1_18 = "E1.18"  # Surds (Number - extended)
    E1_2 = "E1.2"  # Sets (Number - extended)
    E1_3 = "E1.3"  # Powers and roots (Number - extended)
    E1_4 = "E1.4"  # Fractions, decimals and percentages (Number - extended)
    E1_5 = "E1.5"  # Ordering (Number - extended)
    E1_6 = "E1.6"  # The four operations (Number - extended)
    E1_7 = "E1.7"  # Indices I (Number - extended)
    E1_8 = "E1.8"  # Standard form (Number - extended)
    E1_9 = "E1.9"  # Estimation (Number - extended)
    E2_1 = "E2.1"  # Introduction to algebra (Algebra and graphs - extended)
    E2_10 = "E2.10"  # Graphs of functions (Algebra and graphs - extended)
    E2_11 = "E2.11"  # Sketching curves (Algebra and graphs - extended)
    E2_12 = "E2.12"  # Differentiation (Algebra and graphs - extended)
    E2_13 = "E2.13"  # Functions (Algebra and graphs - extended)
    E2_2 = "E2.2"  # Algebraic manipulation (Algebra and graphs - extended)
    E2_3 = "E2.3"  # Algebraic fractions (Algebra and graphs - extended)
    E2_4 = "E2.4"  # Indices II (Algebra and graphs - extended)
    E2_5 = "E2.5"  # Equations (Algebra and graphs - extended)
    E2_6 = "E2.6"  # Inequalities (Algebra and graphs - extended)
    E2_7 = "E2.7"  # Sequences (Algebra and graphs - extended)
    E2_8 = "E2.8"  # Proportion (Algebra and graphs - extended)
    E2_9 = "E2.9"  # Graphs in practical situations (Algebra and graphs - extended)
    E3_1 = "E3.1"  # Coordinates (Coordinate geometry - extended)
    E3_2 = "E3.2"  # Drawing linear graphs (Coordinate geometry - extended)
    E3_3 = "E3.3"  # Gradient of linear graphs (Coordinate geometry - extended)
    E3_4 = "E3.4"  # Length and midpoint (Coordinate geometry - extended)
    E3_5 = "E3.5"  # Equations of linear graphs (Coordinate geometry - extended)
    E3_6 = "E3.6"  # Parallel lines (Coordinate geometry - extended)
    E3_7 = "E3.7"  # Perpendicular lines (Coordinate geometry - extended)
    E4_1 = "E4.1"  # Geometrical terms (Geometry - extended)
    E4_2 = "E4.2"  # Geometrical constructions (Geometry - extended)
    E4_3 = "E4.3"  # Scale drawings (Geometry - extended)
    E4_4 = "E4.4"  # Similarity (Geometry - extended)
    E4_5 = "E4.5"  # Symmetry (Geometry - extended)
    E4_6 = "E4.6"  # Angles (Geometry - extended)
    E4_7 = "E4.7"  # Circle theorems I (Geometry - extended)
    E4_8 = "E4.8"  # Circle theorems II (Geometry - extended)
    E5_1 = "E5.1"  # Units of measure (Mensuration - extended)
    E5_2 = "E5.2"  # Area and perimeter (Mensuration - extended)
    E5_3 = "E5.3"  # Circles, arcs and sectors (Mensuration - extended)
    E5_4 = "E5.4"  # Surface area and volume (Mensuration - extended)
    E5_5 = "E5.5"  # Compound shapes and parts of shapes (Mensuration - extended)
    E6_1 = "E6.1"  # Pythagoras' theorem (Trigonometry - extended)
    E6_2 = "E6.2"  # Right-angled triangles (Trigonometry - extended)
    E6_3 = "E6.3"  # Exact trigonometric values (Trigonometry - extended)
    E6_4 = "E6.4"  # Trigonometric functions (Trigonometry - extended)
    E6_5 = "E6.5"  # Non-right-angled triangles (Trigonometry - extended)
    E6_6 = "E6.6"  # Pythagoras' theorem and trigonometry in 3D (Trigonometry - extended)
    E7_1 = "E7.1"  # Transformations (Transformations and vectors - extended)
    E7_2 = "E7.2"  # Vectors in two dimensions (Transformations and vectors - extended)
    E7_3 = "E7.3"  # Magnitude of a vector (Transformations and vectors - extended)
    E7_4 = "E7.4"  # Vector geometry (Transformations and vectors - extended)
    E8_1 = "E8.1"  # Introduction to probability (Probability - extended)
    E8_2 = "E8.2"  # Relative and expected frequencies (Probability - extended)
    E8_3 = "E8.3"  # Probability of combined events (Probability - extended)
    E8_4 = "E8.4"  # Conditional probability (Probability - extended)
    E9_1 = "E9.1"  # Classifying statistical data (Statistics - extended)
    E9_2 = "E9.2"  # Interpreting statistical data (Statistics - extended)
    E9_3 = "E9.3"  # Averages and measures of spread (Statistics - extended)
    E9_4 = "E9.4"  # Statistical charts and diagrams (Statistics - extended)
    E9_5 = "E9.5"  # Scatter diagrams (Statistics - extended)
    E9_6 = "E9.6"  # Cumulative frequency diagrams (Statistics - extended)
    E9_7 = "E9.7"  # Histograms (Statistics - extended)


class TopicPathComponent(Enum):
    """Valid topic path components for Cambridge IGCSE Mathematics"""
    ALGEBRA_AND_GRAPHS = "Algebra and graphs"
    ALGEBRAIC_MANIPULATION = "Algebraic manipulation"
    ANGLES = "Angles"
    ANGLES_ON_A_STRAIGHT_LINE = "Angles on a straight line"
    AREA_AND_PERIMETER = "Area and perimeter"
    AVERAGES_AND_RANGE = "Averages and range"
    BEARINGS = "Bearings"
    COORDINATE_GEOMETRY = "Coordinate geometry"
    EQUATIONS = "Equations"
    EQUATIONS_OF_LINEAR_GRAPHS = "Equations of linear graphs"
    ESTIMATION = "Estimation"
    FRACTIONS_DECIMALS_AND_PERCENTAGES = "Fractions, decimals and percentages"
    GEOMETRICAL_TERMS = "Geometrical terms"
    GEOMETRY = "Geometry"
    GRAPHS_OF_FUNCTIONS = "Graphs of functions"
    INDICES_I = "Indices I"
    INEQUALITIES = "Inequalities"
    LIMITS_OF_ACCURACY = "Limits of accuracy"
    MENSURATION = "Mensuration"
    NETS = "Nets"
    NUMBER = "Number"
    ORDERING = "Ordering"
    PARALLEL_LINES = "Parallel lines"
    PERCENTAGES = "Percentages"
    PLACE_VALUE = "Place value"
    PROBABILITY = "Probability"
    PROBABILITY_OF_COMBINED_EVENTS = "Probability of combined events"
    PROPERTIES_OF_SOLIDS = "Properties of solids"
    RATIO_AND_PROPORTION = "Ratio and proportion"
    SCALE_DRAWINGS = "Scale drawings"
    SCATTER_DIAGRAMS = "Scatter diagrams"
    SETS = "Sets"
    SKETCHING_CURVES = "Sketching curves"
    STATISTICS = "Statistics"
    SYMMETRY = "Symmetry"
    THE_FOUR_OPERATIONS = "The four operations"
    TIME = "Time"
    TRANSFORMATIONS = "Transformations"
    TRANSFORMATIONS_AND_VECTORS = "Transformations and vectors"
    TRIANGLES = "Triangles"
    TYPES_OF_NUMBER = "Types of number"


# Helper functions to convert strings to enums
def skill_tag_from_string(tag_str: str) -> SkillTag:
    """Convert string to SkillTag enum, with fallback"""
    try:
        return SkillTag(tag_str)
    except ValueError:
        # Fallback for unrecognized skill tags
        print(f"Warning: Unrecognized skill tag '{tag_str}', using WORD_PROBLEM as fallback")
        return SkillTag.WORD_PROBLEM


def subject_ref_from_string(ref_str: str) -> SubjectContentReference:
    """Convert string to SubjectContentReference enum, with fallback"""
    try:
        return SubjectContentReference(ref_str)
    except ValueError:
        # Fallback for unrecognized references
        print(f"Warning: Unrecognized subject reference '{ref_str}', using C1_6 as fallback")
        return SubjectContentReference.C1_6


def topic_path_from_string(path_str: str) -> TopicPathComponent:
    """Convert string to TopicPathComponent enum, with fallback"""
    try:
        return TopicPathComponent(path_str)
    except ValueError:
        # Fallback for unrecognized topic paths
        print(f"Warning: Unrecognized topic path '{path_str}', using NUMBER as fallback")
        return TopicPathComponent.NUMBER


# Validation helpers
def get_valid_skill_tags() -> list[str]:
    """Get list of all valid skill tag strings"""
    return [tag.value for tag in SkillTag]


def get_valid_subject_refs() -> list[str]:
    """Get list of all valid subject content reference strings"""
    return [ref.value for ref in SubjectContentReference]


def get_valid_topic_paths() -> list[str]:
    """Get list of all valid topic path component strings"""
    return [path.value for path in TopicPathComponent]
