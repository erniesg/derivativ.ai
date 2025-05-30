"""
Comprehensive Question Validator for Cambridge IGCSE Mathematics.

Validates candidate questions against:
- Cambridge syllabus standards
- Schema compliance
- Data integrity
- Mathematical accuracy
- Marking scheme consistency
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from ..models import CandidateQuestion, CommandWord, GenerationStatus
from ..models.question_models import CalculatorPolicy


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    CRITICAL = "critical"    # Prevents insertion
    WARNING = "warning"      # Allows insertion with note
    INFO = "info"           # Informational only


@dataclass
class ValidationIssue:
    """Individual validation issue"""
    severity: ValidationSeverity
    field: str
    issue_type: str
    message: str
    suggested_fix: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of question validation"""
    is_valid: bool
    issues: List[ValidationIssue]
    warnings_count: int
    critical_errors_count: int

    @property
    def can_insert(self) -> bool:
        """Whether question can be inserted (no critical errors)"""
        return self.critical_errors_count == 0


class CambridgeQuestionValidator:
    """Comprehensive validator for Cambridge IGCSE Mathematics questions"""

    def __init__(self):
        self.syllabus_data = self._load_syllabus_data()
        self.valid_subject_refs = self._extract_valid_subject_refs()
        self.valid_topic_paths = self._extract_valid_topic_paths()
        self.command_words = self._load_command_words()

    def _load_syllabus_data(self) -> Dict[str, Any]:
        """Load Cambridge syllabus data"""
        try:
            with open("data/syllabus_command.json", "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load syllabus data: {e}")
            return {}

    def _extract_valid_subject_refs(self) -> set:
        """Extract all valid subject content references from syllabus"""
        refs = set()

        for content_type in ["core_subject_content", "extended_subject_content"]:
            for topic in self.syllabus_data.get(content_type, []):
                for sub_topic in topic.get("sub_topics", []):
                    ref = sub_topic.get("subject_content_ref")
                    if ref:
                        refs.add(ref)

        return refs

    def _extract_valid_topic_paths(self) -> Dict[str, List[str]]:
        """Extract valid topic paths from syllabus"""
        topic_paths = {}

        for content_type in ["core_subject_content", "extended_subject_content"]:
            for topic in self.syllabus_data.get(content_type, []):
                topic_name = topic.get("topic_name", "")
                if topic_name:
                    topic_paths[topic_name] = []
                    for sub_topic in topic.get("sub_topics", []):
                        sub_name = sub_topic.get("title", "")
                        if sub_name:
                            topic_paths[topic_name].append(sub_name)

        return topic_paths

    def _load_command_words(self) -> set:
        """Load valid command words from syllabus"""
        command_words = set()
        for cw in self.syllabus_data.get("command_words", []):
            word = cw.get("command_word")
            if word:
                command_words.add(word)
        return command_words

    def validate_question(self, question: CandidateQuestion) -> ValidationResult:
        """Comprehensive validation of a candidate question"""
        issues = []

        # 1. Schema and structure validation
        issues.extend(self._validate_schema_compliance(question))

        # 2. Cambridge standards validation
        issues.extend(self._validate_cambridge_standards(question))

        # 3. Subject content reference validation
        issues.extend(self._validate_subject_references(question))

        # 4. Topic path validation
        issues.extend(self._validate_topic_paths(question))

        # 5. Marking scheme validation
        issues.extend(self._validate_marking_scheme(question))

        # 6. Mathematical consistency validation
        issues.extend(self._validate_mathematical_consistency(question))

        # 7. Generation metadata validation
        issues.extend(self._validate_generation_metadata(question))

        # Count issues by severity
        critical_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.CRITICAL)
        warning_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.WARNING)

        return ValidationResult(
            is_valid=critical_count == 0,
            issues=issues,
            warnings_count=warning_count,
            critical_errors_count=critical_count
        )

    def _validate_schema_compliance(self, question: CandidateQuestion) -> List[ValidationIssue]:
        """Validate question follows the required schema"""
        issues = []

        # Check required fields are not empty
        if not question.raw_text_content.strip():
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field="raw_text_content",
                issue_type="empty_field",
                message="Question text cannot be empty",
                suggested_fix="Generate or provide question content"
            ))

        if not question.question_id_local:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field="question_id_local",
                issue_type="empty_field",
                message="Local question ID is required",
                suggested_fix="Generate a unique local ID"
            ))

        if not question.question_id_global:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field="question_id_global",
                issue_type="empty_field",
                message="Global question ID is required",
                suggested_fix="Generate a unique global ID"
            ))

        # Check marks are positive
        if question.marks <= 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field="marks",
                issue_type="invalid_value",
                message=f"Marks must be positive, got {question.marks}",
                suggested_fix="Set marks to a positive integer"
            ))

        # Check answers exist
        if not question.solution_and_marking_scheme.final_answers_summary:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field="final_answers_summary",
                issue_type="empty_collection",
                message="Question must have at least one answer",
                suggested_fix="Add answer summary with expected result"
            ))

        # Check solver steps exist
        if not question.solver_algorithm.steps:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field="solver_algorithm.steps",
                issue_type="empty_collection",
                message="Question must have solution steps",
                suggested_fix="Add step-by-step solution algorithm"
            ))

        return issues

    def _validate_cambridge_standards(self, question: CandidateQuestion) -> List[ValidationIssue]:
        """Validate against Cambridge IGCSE standards"""
        issues = []

        # Validate command word
        if question.command_word.value not in self.command_words:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field="command_word",
                issue_type="invalid_command_word",
                message=f"'{question.command_word.value}' is not a valid Cambridge IGCSE command word",
                suggested_fix=f"Use one of: {', '.join(sorted(self.command_words))}"
            ))

        # Validate grade appropriateness
        if not (1 <= question.target_grade_input <= 9):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field="target_grade_input",
                issue_type="invalid_grade",
                message=f"Target grade {question.target_grade_input} is outside valid range 1-9",
                suggested_fix="Set target grade between 1 and 9"
            ))

        # Validate marks range (Cambridge typically 1-5 marks per question)
        if question.marks > 5:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="marks",
                issue_type="unusual_marks",
                message=f"Question has {question.marks} marks, which is unusually high for IGCSE",
                suggested_fix="Consider breaking into sub-parts or reducing complexity"
            ))

        return issues

    def _validate_subject_references(self, question: CandidateQuestion) -> List[ValidationIssue]:
        """Validate subject content references exist in syllabus"""
        issues = []

        for ref in question.taxonomy.subject_content_references:
            if ref not in self.valid_subject_refs:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    field="taxonomy.subject_content_references",
                    issue_type="invalid_subject_ref",
                    message=f"Subject reference '{ref}' not found in Cambridge syllabus",
                    suggested_fix=f"Use valid references like: {', '.join(list(self.valid_subject_refs)[:10])}..."
                ))

        if not question.taxonomy.subject_content_references:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field="taxonomy.subject_content_references",
                issue_type="empty_collection",
                message="Question must have at least one subject content reference",
                suggested_fix="Add appropriate syllabus references"
            ))

        return issues

    def _validate_topic_paths(self, question: CandidateQuestion) -> List[ValidationIssue]:
        """Validate topic paths make sense"""
        issues = []

        topic_path = question.taxonomy.topic_path

        if not topic_path:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field="taxonomy.topic_path",
                issue_type="empty_collection",
                message="Topic path cannot be empty",
                suggested_fix="Add appropriate topic classification"
            ))
            return issues

        # Check if main topic exists in syllabus
        main_topic = topic_path[0]
        if main_topic not in self.valid_topic_paths:
            # Check for close matches
            available_topics = list(self.valid_topic_paths.keys())
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="taxonomy.topic_path",
                issue_type="unrecognized_topic",
                message=f"Main topic '{main_topic}' not recognized in syllabus",
                suggested_fix=f"Consider using: {', '.join(available_topics)}"
            ))

        # Basic sanity checks
        if len(topic_path) < 2:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="taxonomy.topic_path",
                issue_type="incomplete_path",
                message="Topic path should have at least 2 levels (e.g., ['Number', 'Fractions'])",
                suggested_fix="Add more specific topic classification"
            ))

        return issues

    def _validate_marking_scheme(self, question: CandidateQuestion) -> List[ValidationIssue]:
        """Validate marking scheme consistency"""
        issues = []

        marking_scheme = question.solution_and_marking_scheme

        # Check marks add up
        total_criteria_marks = sum(
            criterion.marks_value for criterion in marking_scheme.mark_allocation_criteria
        )

        if abs(total_criteria_marks - question.marks) > 0.01:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field="mark_allocation_criteria",
                issue_type="marks_mismatch",
                message=f"Criteria marks ({total_criteria_marks}) don't match question marks ({question.marks})",
                suggested_fix="Adjust criteria marks to match total question marks"
            ))

        if abs(marking_scheme.total_marks_for_part - question.marks) > 0.01:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field="total_marks_for_part",
                issue_type="marks_mismatch",
                message=f"Total marks for part ({marking_scheme.total_marks_for_part}) doesn't match question marks ({question.marks})",
                suggested_fix="Set total_marks_for_part to match question marks"
            ))

        # Validate mark codes (Cambridge uses M, A, B, etc.)
        valid_mark_codes = {"M", "A", "B", "FT", "SC", "ISW", "OE", "SOI", "CAO", "DEP", "C", "E", "NA"}
        for criterion in marking_scheme.mark_allocation_criteria:
            if criterion.mark_type_primary and criterion.mark_type_primary not in valid_mark_codes:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="mark_allocation_criteria.mark_type_primary",
                    issue_type="invalid_mark_code",
                    message=f"Mark code '{criterion.mark_type_primary}' is not standard Cambridge format",
                    suggested_fix=f"Use standard codes: {', '.join(sorted(valid_mark_codes))}"
                ))

        return issues

    def _validate_mathematical_consistency(self, question: CandidateQuestion) -> List[ValidationIssue]:
        """Validate mathematical consistency"""
        issues = []

        # Check if answers have appropriate precision
        for answer in question.solution_and_marking_scheme.final_answers_summary:
            if answer.value_numeric is not None:
                # Check for reasonable precision (not too many decimal places)
                str_value = str(answer.value_numeric)
                if '.' in str_value:
                    decimal_places = len(str_value.split('.')[1])
                    if decimal_places > 4:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            field="final_answers_summary.value_numeric",
                            issue_type="excessive_precision",
                            message=f"Answer {answer.value_numeric} has {decimal_places} decimal places",
                            suggested_fix="Consider rounding to appropriate precision for grade level"
                        ))

        # Check cognitive level appropriateness for grade
        if question.taxonomy.cognitive_level:
            grade = question.target_grade_input
            cognitive_level = question.taxonomy.cognitive_level

            # Basic appropriateness check
            if grade <= 3 and cognitive_level in ["Analysis", "ProblemSolving"]:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="taxonomy.cognitive_level",
                    issue_type="inappropriate_difficulty",
                    message=f"Cognitive level '{cognitive_level}' may be too advanced for grade {grade}",
                    suggested_fix="Consider 'Recall' or 'ProceduralFluency' for lower grades"
                ))

        return issues

    def _validate_generation_metadata(self, question: CandidateQuestion) -> List[ValidationIssue]:
        """Validate generation metadata and origins"""
        issues = []

        # Check that generated questions have proper origins
        if question.question_id_global.startswith("gen_"):
            # This is a generated question - validate generation metadata
            if not question.generation_id:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    field="generation_id",
                    issue_type="missing_metadata",
                    message="Generated questions must have generation_id",
                    suggested_fix="Set generation_id from the generation process"
                ))

            if not question.llm_model_used_generation:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    field="llm_model_used_generation",
                    issue_type="missing_metadata",
                    message="Generated questions must specify the LLM model used",
                    suggested_fix="Set llm_model_used_generation to the model that created this question"
                ))

            # Generated questions should have status CANDIDATE initially
            if question.status != GenerationStatus.CANDIDATE:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="status",
                    issue_type="unexpected_status",
                    message=f"New generated questions should have status 'candidate', got '{question.status}'",
                    suggested_fix="Set status to 'candidate' for new questions"
                ))

        # Check template versions are specified
        if not question.prompt_template_version_generation:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="prompt_template_version_generation",
                issue_type="missing_version",
                message="Prompt template version not specified",
                suggested_fix="Set template version for reproducibility"
            ))

        return issues

    def get_validation_summary(self, result: ValidationResult) -> str:
        """Generate human-readable validation summary"""
        if result.is_valid:
            if result.warnings_count == 0:
                return "✅ Question passes all validation checks"
            else:
                return f"✅ Question is valid with {result.warnings_count} warning(s)"
        else:
            return f"❌ Question has {result.critical_errors_count} critical error(s) and {result.warnings_count} warning(s)"

    def print_validation_report(self, result: ValidationResult, question_id: str = "Unknown"):
        """Print detailed validation report"""
        print(f"\n🔍 Validation Report for Question: {question_id}")
        print("=" * 60)
        print(self.get_validation_summary(result))

        if result.issues:
            print(f"\n📋 Issues Found ({len(result.issues)}):")

            # Group by severity
            critical_issues = [i for i in result.issues if i.severity == ValidationSeverity.CRITICAL]
            warning_issues = [i for i in result.issues if i.severity == ValidationSeverity.WARNING]
            info_issues = [i for i in result.issues if i.severity == ValidationSeverity.INFO]

            for severity, issues, icon in [
                (ValidationSeverity.CRITICAL, critical_issues, "🚨"),
                (ValidationSeverity.WARNING, warning_issues, "⚠️"),
                (ValidationSeverity.INFO, info_issues, "ℹ️")
            ]:
                if issues:
                    print(f"\n{icon} {severity.value.upper()} ({len(issues)}):")
                    for issue in issues:
                        print(f"   • {issue.field}: {issue.message}")
                        if issue.suggested_fix:
                            print(f"     → Suggested fix: {issue.suggested_fix}")

        print(f"\n🎯 Can insert to database: {'Yes' if result.can_insert else 'No'}")
        print("=" * 60)


# Convenience function for quick validation
def validate_question(question: CandidateQuestion, verbose: bool = False) -> ValidationResult:
    """Quick validation function"""
    validator = CambridgeQuestionValidator()
    result = validator.validate_question(question)

    if verbose:
        validator.print_validation_report(result, question.question_id_local)

    return result
