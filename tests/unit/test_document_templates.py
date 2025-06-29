
from unittest.mock import MagicMock

import pytest

from src.models.document_models import ArtifactType, ContentBlock, DocumentTemplate
from src.services.content_blocks import AnswerKeyBlock, PracticeQuestionsBlock


# Mock ContentBlock implementations for testing DocumentTemplate logic
class MockTitleBlock(ContentBlock):
    async def render(self, **kwargs) -> str:
        return "Title Content"

class MockLearningObjectivesBlock(ContentBlock):
    async def render(self, **kwargs) -> str:
        return "Learning Objectives Content"

class MockKeyConceptsBlock(ContentBlock):
    async def render(self, **kwargs) -> str:
        return "Key Concepts Content"

class MockWorkedExampleBlock(ContentBlock):
    async def render(self, **kwargs) -> str:
        return "Worked Example Content"

class MockActivityBlock(ContentBlock):
    async def render(self, **kwargs) -> str:
        return "Activity Content"

class MockHomeworkAssignmentBlock(ContentBlock):
    async def render(self, **kwargs) -> str:
        return "Homework Assignment Content"

class MockAssessmentPointerBlock(ContentBlock):
    async def render(self, **kwargs) -> str:
        return "Assessment Pointer Content"

class MockSummaryBlock(ContentBlock):
    async def render(self, **kwargs) -> str:
        return "Summary Content"

class MockGlossaryBlock(ContentBlock):
    async def render(self, **kwargs) -> str:
        return "Glossary Content"


class TestDocumentTemplates:

    @pytest.fixture
    def mock_q_gen_service(self):
        return MagicMock()

    def test_worksheet_template_short_duration(self, mock_q_gen_service):
        """
        Tests that a Worksheet template for a short duration (10-20 min)
        selects the correct content blocks.
        """
        # Define the template logic within the test for now
        # In a real scenario, these would be loaded from a config or database
        template = DocumentTemplate(
            name="Worksheet",
            artifact_type=ArtifactType.WORKSHEET,
            block_configs={
                "short": [
                    {"type": MockTitleBlock},
                    {"type": PracticeQuestionsBlock, "params": {"count": 5, "q_gen_service": mock_q_gen_service}},
                    {"type": AnswerKeyBlock},
                ],
                "medium": [], # Placeholder
                "long": [], # Placeholder
            }
        )

        # Simulate getting blocks for a 15-minute duration
        # This logic will be part of the DocumentTemplate class later
        selected_blocks = []
        if 10 <= 15 <= 20: # Simplified duration check
            for config in template.block_configs["short"]:
                if config["type"] == PracticeQuestionsBlock:
                    selected_blocks.append(config["type"](config["params"]["q_gen_service"]))
                else:
                    selected_blocks.append(config["type"]())

        assert len(selected_blocks) == 3
        assert isinstance(selected_blocks[0], MockTitleBlock)
        assert isinstance(selected_blocks[1], PracticeQuestionsBlock)
        assert isinstance(selected_blocks[2], AnswerKeyBlock)

    def test_worksheet_template_student_copy_excludes_answer_key(self, mock_q_gen_service):
        """
        Tests that a Worksheet template for a student copy excludes the AnswerKeyBlock.
        """
        template = DocumentTemplate(
            name="Worksheet",
            artifact_type=ArtifactType.WORKSHEET,
            block_configs={
                "short": [
                    {"type": MockTitleBlock},
                    {"type": PracticeQuestionsBlock, "params": {"count": 5, "q_gen_service": mock_q_gen_service}},
                    {"type": AnswerKeyBlock},
                ],
            }
        )

        selected_blocks = []
        if 10 <= 15 <= 20: # Simplified duration check
            for config in template.block_configs["short"]:
                if config["type"] == PracticeQuestionsBlock:
                    selected_blocks.append(config["type"](config["params"]["q_gen_service"]))
                else:
                    selected_blocks.append(config["type"]())

        # Simulate student_copy logic (this will be integrated into DocumentTemplate)
        final_blocks = [b for b in selected_blocks if not isinstance(b, AnswerKeyBlock)]

        assert len(final_blocks) == 2
        assert isinstance(final_blocks[0], MockTitleBlock)
        assert isinstance(final_blocks[1], PracticeQuestionsBlock)
        assert not any(isinstance(b, AnswerKeyBlock) for b in final_blocks)
