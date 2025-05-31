"""
Enums for Cambridge IGCSE Mathematics question generation.
Based on actual data from 2025p1.json to ensure consistency.
"""

from enum import Enum


class SkillTag(Enum):
    """Valid skill tags from Cambridge IGCSE Mathematics past papers"""
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
    """Valid Cambridge IGCSE Mathematics subject content references"""
    C1_1 = "C1.1"   # Natural numbers, integers, prime numbers, square numbers, cube numbers
    C1_2 = "C1.2"   # Recognise and use multiples, factors, common factors, highest common factor, lowest common multiple
    C1_4 = "C1.4"   # Fractions, decimals and percentages
    C1_5 = "C1.5"   # Ordering rational numbers
    C1_6 = "C1.6"   # The four operations (+, −, ×, ÷) for integers and decimals
    C1_7 = "C1.7"   # Indices
    C1_9 = "C1.9"   # Standard form
    C1_10 = "C1.10" # Estimates, approximations and limits of accuracy
    C1_11 = "C1.11" # Ratio and proportion
    C1_13 = "C1.13" # Percentages
    C1_15 = "C1.15" # Time, money and other measures
    C2_2 = "C2.2"   # Algebraic manipulation
    C2_5 = "C2.5"   # Linear equations
    C2_6 = "C2.6"   # Linear inequalities
    C2_10 = "C2.10" # Simultaneous linear equations
    C2_11 = "C2.11" # Quadratic equations
    C3_3 = "C3.3"   # Linear graphs
    C3_5 = "C3.5"   # Graphs of functions
    C4_1 = "C4.1"   # Geometrical terms and relationships
    C4_2 = "C4.2"   # Symmetry
    C4_3 = "C4.3"   # Angle properties
    C4_5 = "C4.5"   # Loci
    C4_6 = "C4.6"   # Bearings
    C5_2 = "C5.2"   # Mensuration of 2D and 3D shapes
    C7_1 = "C7.1"   # Transformations
    C8_3 = "C8.3"   # Probability
    C9_3 = "C9.3"   # Measures of central tendency and spread
    C9_5 = "C9.5"   # Scatter diagrams and correlation


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
