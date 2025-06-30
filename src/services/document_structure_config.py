"""
Document structure configuration loader.

Loads document structure patterns from YAML configuration instead of hardcoding them.
"""

import logging
from pathlib import Path

import yaml

from src.models.document_models import DetailLevel, DocumentType

logger = logging.getLogger(__name__)


class DocumentStructureConfig:
    """Loads and manages document structure patterns from configuration."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            # Default to config file in project root
            config_path = (
                Path(__file__).parent.parent.parent / "config" / "document_structures.yaml"
            )

        self.config_path = Path(config_path)
        self._structure_patterns = None
        self._load_config()

    def _load_config(self):
        """Load structure patterns from YAML configuration."""
        try:
            with open(self.config_path) as f:
                config_data = yaml.safe_load(f)

            # Convert string keys to proper enums and ints
            self._structure_patterns = {}

            for doc_type_str, detail_patterns in config_data["document_structures"].items():
                doc_type = DocumentType(doc_type_str)
                self._structure_patterns[doc_type] = {}

                for detail_level_int, pattern_list in detail_patterns.items():
                    # Convert integer detail level to DetailLevel enum
                    detail_level = self._int_to_detail_level(detail_level_int)
                    self._structure_patterns[doc_type][detail_level] = pattern_list

            logger.info(f"Loaded document structures from {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to load document structures from {self.config_path}: {e}")
            # Fallback to minimal hardcoded patterns
            self._structure_patterns = self._get_fallback_patterns()

    def _int_to_detail_level(self, level_int: int) -> DetailLevel:
        """Convert integer to closest DetailLevel enum."""
        if level_int <= 1:
            return DetailLevel.MINIMAL
        elif level_int <= 3:
            return DetailLevel.BASIC
        elif level_int <= 5:
            return DetailLevel.MEDIUM
        elif level_int <= 7:
            return DetailLevel.DETAILED
        elif level_int <= 9:
            return DetailLevel.COMPREHENSIVE
        else:
            return DetailLevel.GUIDED

    def _get_fallback_patterns(self) -> dict[DocumentType, dict[DetailLevel, list[str]]]:
        """Fallback patterns if config loading fails."""
        return {
            DocumentType.WORKSHEET: {
                DetailLevel.MINIMAL: ["practice_questions", "answers"],
                DetailLevel.MEDIUM: [
                    "learning_objectives",
                    "worked_examples",
                    "practice_questions",
                    "solutions",
                ],
                DetailLevel.COMPREHENSIVE: [
                    "learning_objectives",
                    "key_concepts",
                    "worked_examples",
                    "practice_questions",
                    "detailed_solutions",
                ],
            },
            DocumentType.NOTES: {
                DetailLevel.MINIMAL: ["key_concepts", "examples"],
                DetailLevel.MEDIUM: [
                    "learning_objectives",
                    "concept_explanations",
                    "examples",
                    "summary",
                ],
                DetailLevel.COMPREHENSIVE: [
                    "learning_objectives",
                    "detailed_theory",
                    "examples",
                    "applications",
                    "summary",
                ],
            },
        }

    def get_structure_pattern(
        self, document_type: DocumentType, detail_level: DetailLevel
    ) -> list[str]:
        """Get structure pattern for given document type and detail level."""
        try:
            return self._structure_patterns[document_type][detail_level]
        except KeyError:
            detail_level_value = (
                detail_level.value if hasattr(detail_level, "value") else detail_level
            )
            logger.warning(
                f"No structure pattern found for {document_type.value} at {detail_level_value}, using fallback"
            )
            # Fallback to medium level for the document type
            fallback_patterns = self._structure_patterns.get(document_type, {})
            return fallback_patterns.get(
                DetailLevel.MEDIUM, ["learning_objectives", "content", "summary"]
            )

    def get_all_patterns(self) -> dict[DocumentType, dict[DetailLevel, list[str]]]:
        """Get all structure patterns."""
        return self._structure_patterns.copy()

    def reload_config(self):
        """Reload configuration from file."""
        self._load_config()


# Global instance for easy access
_config_instance = None


def get_document_structure_config() -> DocumentStructureConfig:
    """Get global document structure configuration instance."""
    global _config_instance  # noqa: PLW0603
    if _config_instance is None:
        _config_instance = DocumentStructureConfig()
    return _config_instance
