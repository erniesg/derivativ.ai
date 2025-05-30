# Validation module
"""
Validation services for Cambridge IGCSE Mathematics questions.
Ensures questions meet Cambridge standards before database insertion.
"""

from .question_validator import (
    CambridgeQuestionValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    validate_question
)

__all__ = [
    "CambridgeQuestionValidator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    "validate_question"
]
