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
from ..models.enums import (
    SkillTag, SubjectContentReference, TopicPathComponent,
    get_valid_skill_tags, get_valid_subject_refs, get_valid_topic_paths
)


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
        # Use enum-based validation for consistency
        self.valid_skill_tags = set(get_valid_skill_tags())
        self.valid_subject_refs = set(get_valid_subject_refs())
        self.valid_topic_paths = set(get_valid_topic_paths())

        # Still load syllabus data for additional context
        self.syllabus_data = self._load_syllabus_data()
        self.command_words = self._load_command_words()

    def _load_syllabus_data(self) -> Dict[str, Any]:
        """Load Cambridge syllabus data"""
        try:
            with open("data/syllabus_command.json", "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load syllabus data: {e}")
            return {}

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

        # 3. Subject content reference validation (using enums)
        issues.extend(self._validate_subject_references(question))

        # 4. Topic path validation (using enums)
        issues.extend(self._validate_topic_paths(question))

        # 5. Skill tags validation (using enums)
        issues.extend(self._validate_skill_tags(question))

        # 6. Marking scheme validation
        issues.extend(self._validate_marking_scheme(question))

        # 7. Mathematical consistency validation
        issues.extend(self._validate_mathematical_consistency(question))

        # 8. Generation metadata validation
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
        """Validate adherence to Cambridge IGCSE standards"""
        issues = []

        # Check target grade is within valid range
        if not (1 <= question.target_grade_input <= 9):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field="target_grade_input",
                issue_type="invalid_range",
                message=f"Target grade {question.target_grade_input} outside valid range 1-9",
                suggested_fix="Set target grade between 1 and 9"
            ))

        # Check command word is valid
        if question.command_word:
            if question.command_word.value not in self.command_words and self.command_words:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="command_word",
                    issue_type="invalid_command_word",
                    message=f"Command word '{question.command_word.value}' not in Cambridge syllabus",
                    suggested_fix=f"Use standard command words: {', '.join(list(self.command_words)[:5])}..."
                ))

        # Check marks align with typical Cambridge ranges
        if question.marks > 10:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="marks",
                issue_type="unusual_value",
                message=f"Question worth {question.marks} marks is unusually high for IGCSE",
                suggested_fix="Consider breaking into sub-questions or reducing marks"
            ))

        return issues

    def _validate_subject_references(self, question: CandidateQuestion) -> List[ValidationIssue]:
        """Validate subject content references using enums"""
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
        """Validate topic paths using enums"""
        issues = []

        if not question.taxonomy.topic_path:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field="taxonomy.topic_path",
                issue_type="empty_collection",
                message="Question must have topic path",
                suggested_fix="Add topic classification path"
            ))
        else:
            for path_component in question.taxonomy.topic_path:
                if path_component not in self.valid_topic_paths:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        field="taxonomy.topic_path",
                        issue_type="invalid_topic_path",
                        message=f"Topic path component '{path_component}' not in standard Cambridge structure",
                        suggested_fix=f"Use standard paths like: {', '.join(list(self.valid_topic_paths)[:10])}..."
                    ))

            # Check path makes sense (at least 2 levels)
            if len(question.taxonomy.topic_path) < 2:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="taxonomy.topic_path",
                    issue_type="insufficient_depth",
                    message="Topic path should have at least 2 levels (e.g., ['Number', 'Fractions'])",
                    suggested_fix="Add more specific topic classification"
                ))

        return issues

    def _validate_skill_tags(self, question: CandidateQuestion) -> List[ValidationIssue]:
        """Validate skill tags using enums"""
        issues = []

        if not question.taxonomy.skill_tags:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field="taxonomy.skill_tags",
                issue_type="empty_collection",
                message="Question must have at least one skill tag",
                suggested_fix="Add relevant skill tags for the question"
            ))
        else:
            for skill_tag in question.taxonomy.skill_tags:
                if skill_tag not in self.valid_skill_tags:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        field="taxonomy.skill_tags",
                        issue_type="invalid_skill_tag",
                        message=f"Skill tag '{skill_tag}' not in standardized skill taxonomy",
                        suggested_fix=f"Use standard skills like: {', '.join(list(self.valid_skill_tags)[:10])}..."
                    ))

            # Check minimum number of skill tags
            if len(question.taxonomy.skill_tags) < 2:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="taxonomy.skill_tags",
                    issue_type="insufficient_tags",
                    message="Consider adding more skill tags for better classification",
                    suggested_fix="Add at least 2-3 relevant skill tags"
                ))

        return issues

    def _validate_marking_scheme(self, question: CandidateQuestion) -> List[ValidationIssue]:
        """Validate marking scheme consistency"""
        issues = []

        marking_scheme = question.solution_and_marking_scheme

        # Check total marks consistency
        if marking_scheme.total_marks_for_part != question.marks:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field="solution_and_marking_scheme.total_marks_for_part",
                issue_type="inconsistent_marks",
                message=f"Total marks ({marking_scheme.total_marks_for_part}) doesn't match question marks ({question.marks})",
                suggested_fix="Ensure marking scheme total equals question marks"
            ))

        # Check criteria marks sum to total
        criteria_total = sum(criterion.marks_value for criterion in marking_scheme.mark_allocation_criteria)
        if abs(criteria_total - marking_scheme.total_marks_for_part) > 0.01:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field="mark_allocation_criteria",
                issue_type="marks_mismatch",
                message=f"Mark criteria total ({criteria_total}) doesn't match total marks ({marking_scheme.total_marks_for_part})",
                suggested_fix="Adjust individual criteria marks to sum correctly"
            ))

        # Check each criterion has valid details
        for i, criterion in enumerate(marking_scheme.mark_allocation_criteria):
            if not criterion.criterion_text.strip():
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field=f"mark_allocation_criteria[{i}].criterion_text",
                    issue_type="empty_field",
                    message=f"Mark criterion {i+1} has empty description",
                    suggested_fix="Add descriptive text for what earns the marks"
                ))

            if criterion.marks_value <= 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    field=f"mark_allocation_criteria[{i}].marks_value",
                    issue_type="invalid_value",
                    message=f"Mark criterion {i+1} has non-positive marks ({criterion.marks_value})",
                    suggested_fix="Set positive mark value"
                ))

        return issues

    def _validate_mathematical_consistency(self, question: CandidateQuestion) -> List[ValidationIssue]:
        """Validate mathematical consistency and accuracy"""
        issues = []

        # Check that solver algorithm has reasonable steps
        if len(question.solver_algorithm.steps) == 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field="solver_algorithm.steps",
                issue_type="empty_collection",
                message="Solution algorithm must have at least one step",
                suggested_fix="Add step-by-step solution process"
            ))

        # Check answers have appropriate format
        for i, answer in enumerate(question.solution_and_marking_scheme.final_answers_summary):
            if not answer.answer_text.strip():
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    field=f"final_answers_summary[{i}].answer_text",
                    issue_type="empty_field",
                    message=f"Answer {i+1} has empty text",
                    suggested_fix="Provide clear answer text"
                ))

        return issues

    def _validate_generation_metadata(self, question: CandidateQuestion) -> List[ValidationIssue]:
        """Validate generation metadata and traceability"""
        issues = []

        if not question.generation_id:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="generation_id",
                issue_type="missing_metadata",
                message="Generation ID missing - reduces traceability",
                suggested_fix="Include generation ID for audit trail"
            ))

        if not question.llm_model_used_generation:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="llm_model_used_generation",
                issue_type="missing_metadata",
                message="LLM model information missing",
                suggested_fix="Record which model generated the question"
            ))

        return issues

    def get_validation_summary(self, result: ValidationResult) -> str:
        """Get a brief validation summary"""
        if result.is_valid:
            return f"✅ Valid ({result.warnings_count} warnings)"
        else:
            return f"❌ Invalid ({result.critical_errors_count} critical errors, {result.warnings_count} warnings)"

    def print_validation_report(self, result: ValidationResult, question_id: str = "Unknown"):
        """Print detailed validation report"""
        print(f"\n📋 Validation Report for Question {question_id}")
        print("=" * 60)

        if result.is_valid:
            print("✅ VALIDATION PASSED")
        else:
            print("❌ VALIDATION FAILED")

        print(f"Critical Errors: {result.critical_errors_count}")
        print(f"Warnings: {result.warnings_count}")

        if result.issues:
            print("\n🔍 Issues Found:")
            for issue in result.issues:
                icon = "🚨" if issue.severity == ValidationSeverity.CRITICAL else "⚠️"
                print(f"{icon} {issue.severity.value.upper()}: {issue.field}")
                print(f"   {issue.message}")
                if issue.suggested_fix:
                    print(f"   💡 Fix: {issue.suggested_fix}")
                print()


def validate_question(question: CandidateQuestion, verbose: bool = False) -> ValidationResult:
    """Convenience function to validate a question"""
    validator = CambridgeQuestionValidator()
    result = validator.validate_question(question)

    if verbose:
        validator.print_validation_report(result, question.question_id_global or "Unknown")

    return result
