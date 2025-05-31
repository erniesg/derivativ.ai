"""
Skill Tag Mapper Service - Maps subject content references to relevant skill tags.

This service provides contextually appropriate skill tags based on the subject
content references, solving the dependency between topics and skills.
"""

from typing import List, Set
from ..models.enums import get_valid_skill_tags, get_valid_subject_refs


class SkillTagMapper:
    """Maps subject content references to contextually relevant skill tags"""

    def __init__(self):
        self.all_skill_tags = set(get_valid_skill_tags())
        self.all_subject_refs = set(get_valid_subject_refs())

        # Create mapping from subject content references to relevant skill tags
        self._initialize_mappings()

    def _initialize_mappings(self):
        """Initialize the mapping from subject content references to skill tags"""

        # Number topics (C1.x and E1.x)
        self.number_skills = {
            "ADDITION", "SUBTRACTION", "MULTIPLICATION", "DIVISION",
            "PLACE_VALUE", "ROUNDING", "ORDER_OF_OPERATIONS",
            "FRACTION_OF_QUANTITY", "PERCENTAGES_SUM_TO_100",
            "PRIME_FACTORIZATION", "HCF", "MULTIPLE", "LOWEST_COMMON_MULTIPLE_CONCEPT",
            "TIME_CALCULATION", "TIME_ZONES", "DURATION",
            "FORMING_NUMBERS", "EVEN_NUMBERS", "NUMBER_RANGE", "NUMBER_TO_WORDS",
            "NEAREST_HUNDRED", "SQUARE_NUMBER", "NEGATIVE_INDICES",
            "UPPER_BOUND", "LOWER_BOUND", "LIMITS_OF_ACCURACY",
            "RATIO_SIMPLIFICATION", "SUBTRACTION_DECIMALS",
            "SUBTRACTION_NEGATIVE_NUMBERS", "SUBTRACTION_WHOLE_NUMBERS",
            "WORD_PROBLEM", "LOGICAL_REASONING"
        }

        # Algebra topics (C2.x and E2.x)
        self.algebra_skills = {
            "SUBSTITUTION", "SOLVE_LINEAR_EQUATION", "FORM_EQUATION",
            "FACTORISATION_COMMON_FACTOR", "SIMULTANEOUS_LINEAR_EQUATIONS",
            "ELIMINATION_METHOD", "SUBSTITUTION_METHOD",
            "INEQUALITIES", "INEQUALITY_FROM_NUMBER_LINE", "INTEGER_SOLUTIONS",
            "QUADRATIC_FUNCTION", "QUADRATIC_GRAPH", "LINE_OF_SYMMETRY_QUADRATIC",
            "EQUATION_OF_A_LINE", "GRADIENT", "Y_INTERCEPT",
            "INTERSECTION_OF_GRAPHS", "REARRANGE_FORMULA",
            "TABLE_OF_VALUES", "PLOTTING_GRAPH", "PLOTTING_POINTS",
            "READING_FROM_GRAPH", "NEGATIVE_INDICES",
            "WORD_PROBLEM", "LOGICAL_REASONING"
        }

        # Coordinate geometry topics (C3.x and E3.x)
        self.coordinate_skills = {
            "PLOTTING_POINTS", "PLOTTING_GRAPH", "GRADIENT",
            "EQUATION_OF_A_LINE", "Y_INTERCEPT", "INTERSECTION_OF_GRAPHS",
            "READING_FROM_GRAPH", "TABLE_OF_VALUES",
            "WORD_PROBLEM", "LOGICAL_REASONING"
        }

        # Geometry topics (C4.x and E4.x)
        self.geometry_skills = {
            "ANGLE_PROPERTIES", "ANGLES_IN_A_TRIANGLE", "ALTERNATE_ANGLES",
            "ANGLES_ON_A_STRAIGHT_LINE", "CONSTRUCTION", "MEASUREMENT_ANGLE",
            "MEASUREMENT_LENGTH", "BEARINGS", "SCALE_DRAWING", "SCALE_FACTOR",
            "AREA_COMPOSITE_SHAPES", "AREA_PARALLELOGRAM",
            "ROTATIONAL_SYMMETRY", "ORDER_OF_ROTATIONAL_SYMMETRY",
            "IDENTIFY_SOLID_FROM_NET", "VERTICES_OF_SOLID", "SQUARE_BASED_PYRAMID",
            "WORD_PROBLEM", "VISUALIZATION"
        }

        # Mensuration topics (C5.x and E5.x)
        self.mensuration_skills = {
            "AREA_COMPOSITE_SHAPES", "AREA_PARALLELOGRAM", "MEASUREMENT_LENGTH",
            "SCALE_FACTOR", "VISUALIZATION", "IDENTIFY_SOLID_FROM_NET",
            "VERTICES_OF_SOLID", "SQUARE_BASED_PYRAMID",
            "WORD_PROBLEM", "LOGICAL_REASONING"
        }

        # Trigonometry topics (C6.x and E6.x)
        self.trigonometry_skills = {
            "MEASUREMENT_ANGLE", "MEASUREMENT_LENGTH", "ANGLES_IN_A_TRIANGLE",
            "WORD_PROBLEM", "LOGICAL_REASONING", "VISUALIZATION"
        }

        # Transformations topics (C7.x and E7.x)
        self.transformation_skills = {
            "ROTATION", "TRANSLATION", "ENLARGEMENT", "DESCRIBE_TRANSFORMATION",
            "SCALE_FACTOR", "CENTER_OF_ROTATION", "CENTRE_OF_ENLARGEMENT",
            "ANGLE_OF_ROTATION", "DIRECTION_OF_ROTATION",
            "VECTOR_NOTATION", "ROTATIONAL_SYMMETRY",
            "ORDER_OF_ROTATIONAL_SYMMETRY",
            "WORD_PROBLEM", "VISUALIZATION"
        }

        # Probability topics (C8.x and E8.x)
        self.probability_skills = {
            "PROBABILITY_COMPLEMENT", "PROBABILITY_AND_RULE",
            "TREE_DIAGRAM_COMPLETION", "TREE_DIAGRAM_USE",
            "WORD_PROBLEM", "LOGICAL_REASONING", "ORDERING_DATA"
        }

        # Statistics topics (C9.x and E9.x)
        self.statistics_skills = {
            "MODE", "MEDIAN", "UNGROUPED_DATA", "ORDERING_DATA",
            "SCATTER_DIAGRAM", "SCATTER_DIAGRAM_INTERPRETATION",
            "CORRELATION_TYPE", "LINE_OF_BEST_FIT_DRAWING",
            "ESTIMATION_USING_LINE_OF_BEST_FIT", "READING_FROM_GRAPH",
            "SET_OPERATIONS", "SET_INTERSECTION", "SET_COMPLEMENT_CARDINALITY",
            "VENN_DIAGRAM_COMPLETION",
            "WORD_PROBLEM", "LOGICAL_REASONING"
        }

    def get_relevant_skill_tags(self, subject_content_references: List[str]) -> List[str]:
        """
        Get skill tags relevant to the given subject content references.

        Args:
            subject_content_references: List of subject content refs (e.g., ["C2.2", "C2.5"])

        Returns:
            List of relevant skill tags for those topics
        """
        relevant_skills = set()

        for ref in subject_content_references:
            if not ref:
                continue

            # Extract the topic area from the reference
            if ref.startswith(('C1.', 'E1.')):  # Number
                relevant_skills.update(self.number_skills)
            elif ref.startswith(('C2.', 'E2.')):  # Algebra and graphs
                relevant_skills.update(self.algebra_skills)
            elif ref.startswith(('C3.', 'E3.')):  # Coordinate geometry
                relevant_skills.update(self.coordinate_skills)
            elif ref.startswith(('C4.', 'E4.')):  # Geometry
                relevant_skills.update(self.geometry_skills)
            elif ref.startswith(('C5.', 'E5.')):  # Mensuration
                relevant_skills.update(self.mensuration_skills)
            elif ref.startswith(('C6.', 'E6.')):  # Trigonometry
                relevant_skills.update(self.trigonometry_skills)
            elif ref.startswith(('C7.', 'E7.')):  # Transformations and vectors
                relevant_skills.update(self.transformation_skills)
            elif ref.startswith(('C8.', 'E8.')):  # Probability
                relevant_skills.update(self.probability_skills)
            elif ref.startswith(('C9.', 'E9.')):  # Statistics
                relevant_skills.update(self.statistics_skills)

        # Filter to only include valid skill tags and return sorted list
        valid_relevant_skills = [tag for tag in relevant_skills if tag in self.all_skill_tags]
        return sorted(valid_relevant_skills)

    def get_topic_specific_mappings(self) -> dict:
        """
        Get the complete mapping for inspection/debugging.

        Returns:
            Dictionary mapping topic areas to their skill tags
        """
        return {
            "Number": sorted(self.number_skills),
            "Algebra": sorted(self.algebra_skills),
            "Coordinate Geometry": sorted(self.coordinate_skills),
            "Geometry": sorted(self.geometry_skills),
            "Mensuration": sorted(self.mensuration_skills),
            "Trigonometry": sorted(self.trigonometry_skills),
            "Transformations": sorted(self.transformation_skills),
            "Probability": sorted(self.probability_skills),
            "Statistics": sorted(self.statistics_skills)
        }

    def validate_skill_tags_for_subject_refs(
        self,
        skill_tags: List[str],
        subject_content_references: List[str]
    ) -> tuple[List[str], List[str]]:
        """
        Validate that skill tags are appropriate for the subject content references.

        Args:
            skill_tags: List of skill tags to validate
            subject_content_references: List of subject content refs

        Returns:
            Tuple of (valid_tags, invalid_tags)
        """
        relevant_skills = set(self.get_relevant_skill_tags(subject_content_references))

        valid_tags = [tag for tag in skill_tags if tag in relevant_skills]
        invalid_tags = [tag for tag in skill_tags if tag not in relevant_skills]

        return valid_tags, invalid_tags
