"""
Unit tests for PromptManager integration with new LLM models.
Tests written first following TDD approach.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.models.llm_models import LLMRequest, LLMResponse

# Import existing PromptManager and new models
from src.services.prompt_manager import PromptConfig, PromptError, PromptManager


class TestPromptManagerLLMIntegration:
    """Test integration between PromptManager and new LLM models."""

    @pytest.fixture
    def mock_template_files(self):
        """Mock template files for testing."""
        template_content = {
            "question_generator.jinja2": """Generate a {{question_type}} question about {{topic}}.

System: You are a {{grade}} grade {{subject}} teacher.
Difficulty: {{difficulty}}

Requirements:
- Include {{num_parts}} parts
- Use {{assessment_type}} format
- Follow {{curriculum}} standards""",
            "marker_agent.jinja2": """Create a marking scheme for this question:

{{question_text}}

Requirements:
- Total marks: {{total_marks}}
- Mark types: {{mark_types}}
- Grade level: {{grade}}""",
            "system_prompts.jinja2": """You are an expert {{subject}} teacher specializing in {{curriculum}}.

Your expertise includes:
- {{grade}} level content
- {{assessment_type}} design
- {{pedagogical_approach}} methodology

Context: {{context}}
Instructions: {{instructions}}""",
        }
        return template_content

    @pytest.fixture
    def prompt_manager(self, mock_template_files):
        """Create PromptManager with mocked template files."""
        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=True
        ), patch("pathlib.Path.iterdir") as mock_iterdir:
            # Mock template files
            mock_files = []
            for filename, content in mock_template_files.items():
                mock_file = MagicMock()
                mock_file.name = filename
                mock_file.suffix = ".jinja2"
                mock_file.is_file.return_value = True
                mock_file.read_text.return_value = content
                mock_files.append(mock_file)

            mock_iterdir.return_value = mock_files

            manager = PromptManager(templates_dir="mock_templates")
            return manager

    @pytest.mark.asyncio
    async def test_create_llm_request_from_template(self, prompt_manager):
        """Test creating LLM request using template with variables."""
        variables = {
            "topic": "quadratic equations",
            "target_grade": 10,
            "marks": 4,
            "calculator_policy": "allowed",
            "command_word": "Calculate",
            "subject_content_references": ["C2.5", "C2.10"],
            "include_diagrams": False,
            "cognitive_level": "Application",
            "tier": "Core",
        }

        # Get rendered template
        rendered_content = await prompt_manager.render_prompt(
            PromptConfig(template_name="question_generation", variables=variables)
        )

        # Create LLM request with rendered content
        request = LLMRequest(
            model="gpt-4.1-nano",
            prompt=rendered_content,
            temperature=0.7,
            system_message="You are a mathematics teacher",
            stream=False,
        )

        # Verify template variables were substituted
        assert "quadratic equations" in request.prompt
        assert "Target Grade: 10" in request.prompt
        assert "Marks: 4" in request.prompt
        assert "Calculator Policy: allowed" in request.prompt
        assert "Command Word: Calculate" in request.prompt
        assert "C2.5" in request.prompt and "C2.10" in request.prompt

        # Verify request structure
        assert request.model == "gpt-4.1-nano"
        assert request.temperature == 0.7

    @pytest.mark.asyncio
    async def test_create_llm_request_with_system_prompt_template(self, prompt_manager):
        """Test creating LLM request with system message from template."""
        variables = {
            "subject": "physics",
            "curriculum": "Cambridge IGCSE",
            "grade": "9th",
            "assessment_type": "practical",
            "pedagogical_approach": "inquiry-based",
            "context": "laboratory experiments",
            "instructions": "focus on measurement and uncertainty",
        }

        # Get system prompt from template
        system_prompt = await prompt_manager.render_prompt(
            PromptConfig(template_name="system_prompts", variables=variables)
        )

        # Create LLM request
        request = LLMRequest(
            model="claude-3-5-haiku-20241022",
            prompt="Generate a lab question about forces",
            system_message=system_prompt,
            temperature=0.5,
            extra_params={"top_k": 40},
            stream=False,
        )

        # Verify system message template substitution
        assert "expert physics teacher" in request.system_message
        assert "Cambridge IGCSE" in request.system_message
        assert "9th level content" in request.system_message
        assert "practical design" in request.system_message
        assert "inquiry-based methodology" in request.system_message
        assert "laboratory experiments" in request.system_message
        assert "measurement and uncertainty" in request.system_message

        # Verify request structure
        assert request.model == "claude-3-5-haiku-20241022"
        assert request.temperature == 0.5
        assert request.extra_params["top_k"] == 40

    @pytest.mark.asyncio
    async def test_template_with_missing_variables(self, prompt_manager):
        """Test template rendering with missing variables."""
        variables = {
            "topic": "photosynthesis",
            # Missing required variables: target_grade, marks, calculator_policy
        }

        # Should raise error for missing required variables
        with pytest.raises(PromptError):  # PromptError for missing variables
            await prompt_manager.render_prompt(
                PromptConfig(template_name="question_generation", variables=variables)
            )

    @pytest.mark.asyncio
    async def test_create_request_for_marker_agent(self, prompt_manager):
        """Test creating LLM request for marker agent using template."""
        variables = {
            "question_text": "Calculate the area of a circle with radius 5cm. Show your working.",
            "total_marks": 4,
            "target_grade": 8,
        }

        rendered_prompt = await prompt_manager.render_prompt(
            PromptConfig(template_name="marking_scheme", variables=variables)
        )

        request = LLMRequest(
            model="gpt-4.1-nano",
            prompt=rendered_prompt,
            temperature=0.3,  # Lower temperature for marking consistency
            system_message="You are an experienced examiner",
            stream=False,
        )

        # Verify marking-specific content
        assert "Calculate the area of a circle" in request.prompt
        assert "Show your working" in request.prompt
        assert "**Total Marks:** 4" in request.prompt
        assert "**Grade Level:** 8" in request.prompt

        # Verify appropriate temperature for marking
        assert request.temperature == 0.3

    def test_template_variables_in_extra_params(self, prompt_manager):
        """Test using template variables in extra_params."""
        variables = {
            "thinking_budget": "1024",
            "response_format": "json",
            "custom_instruction": "include step-by-step reasoning",
        }

        # Create request with templated extra_params
        request = LLMRequest(
            model="claude-3-5-haiku-20241022",
            prompt="Generate a question",
            extra_params={
                "thinking": {
                    "budget_tokens": int(variables["thinking_budget"]),
                    "instructions": variables["custom_instruction"],
                },
                "response_mime_type": f"application/{variables['response_format']}",
            },
            stream=False,
        )

        # Verify extra_params use template variables
        assert request.extra_params["thinking"]["budget_tokens"] == 1024
        assert request.extra_params["thinking"]["instructions"] == "include step-by-step reasoning"
        assert request.extra_params["response_mime_type"] == "application/json"

    @pytest.mark.asyncio
    async def test_prompt_manager_with_llm_response_context(self, prompt_manager):
        """Test using LLM response content in subsequent template calls."""
        # Simulate first LLM response
        first_response = LLMResponse(
            content="What is the value of x in the equation 2x + 5 = 13?",
            model_used="gpt-4.1-nano",
            tokens_used=25,
            cost_estimate=0.000012,
            latency_ms=800,
            provider="openai",
        )

        # Use response content in next template
        variables = {
            "question_text": first_response.content,
            "total_marks": 3,
            "target_grade": 7,
        }

        marker_prompt = await prompt_manager.render_prompt(
            PromptConfig(template_name="marking_scheme", variables=variables)
        )

        marker_request = LLMRequest(
            model="claude-3-5-haiku-20241022", prompt=marker_prompt, temperature=0.2, stream=False
        )

        # Verify chaining works
        assert first_response.content in marker_request.prompt
        assert "value of x in the equation 2x + 5 = 13" in marker_request.prompt
        assert "**Total Marks:** 3" in marker_request.prompt

    @pytest.mark.asyncio
    async def test_bulk_request_creation_from_templates(self, prompt_manager):
        """Test creating multiple LLM requests from templates efficiently."""
        # Test data for bulk question generation
        question_configs = [
            {
                "template": "question_generation",
                "variables": {
                    "topic": "algebra",
                    "target_grade": 9,
                    "marks": 2,
                    "calculator_policy": "not_allowed",
                    "command_word": "Solve",
                    "include_diagrams": False,
                    "cognitive_level": "ProceduralFluency",
                    "tier": "Core",
                },
                "model": "gpt-4.1-nano",
                "temperature": 0.8,
            },
            {
                "template": "question_generation",
                "variables": {
                    "topic": "geometry",
                    "target_grade": 10,
                    "marks": 3,
                    "calculator_policy": "allowed",
                    "command_word": "Calculate",
                    "include_diagrams": True,
                    "cognitive_level": "Application",
                    "tier": "Extended",
                },
                "model": "claude-3-5-haiku-20241022",
                "temperature": 0.7,
            },
        ]

        # Create requests in bulk
        requests = []
        for config in question_configs:
            rendered_prompt = await prompt_manager.render_prompt(
                PromptConfig(template_name=config["template"], variables=config["variables"])
            )

            request = LLMRequest(
                model=config["model"],
                prompt=rendered_prompt,
                temperature=config["temperature"],
                stream=False,
            )
            requests.append(request)

        # Verify bulk creation
        assert len(requests) == 2

        # Verify first request
        assert requests[0].model == "gpt-4.1-nano"
        assert requests[0].temperature == 0.8
        assert "algebra" in requests[0].prompt
        assert "Target Grade: 9" in requests[0].prompt

        # Verify second request
        assert requests[1].model == "claude-3-5-haiku-20241022"
        assert requests[1].temperature == 0.7
        assert "geometry" in requests[1].prompt
        assert "Target Grade: 10" in requests[1].prompt

    @pytest.mark.asyncio
    async def test_template_caching_with_llm_requests(self, prompt_manager):
        """Test that template caching works with LLM request creation."""
        variables = {
            "topic": "climate change",
            "target_grade": 11,
            "marks": 5,
            "calculator_policy": "not_allowed",
            "command_word": "Explain",
            "include_diagrams": False,
            "cognitive_level": "ConceptualUnderstanding",
            "tier": "Extended",
        }

        # First call - should load and cache template
        rendered1 = await prompt_manager.render_prompt(
            PromptConfig(template_name="question_generation", variables=variables)
        )
        request1 = LLMRequest(model="gpt-4.1-nano", prompt=rendered1, stream=False)

        # Second call - should use cached template
        rendered2 = await prompt_manager.render_prompt(
            PromptConfig(template_name="question_generation", variables=variables)
        )
        request2 = LLMRequest(model="gpt-4.1-nano", prompt=rendered2, stream=False)

        # Results should be identical
        assert request1.prompt == request2.prompt
        assert "climate change" in request1.prompt
        assert "Target Grade: 11" in request1.prompt

        # Verify caching behavior (implementation detail)
        # This would depend on PromptManager's caching implementation
        assert rendered1 == rendered2


class TestLLMModelTemplateCompatibility:
    """Test that LLM models work seamlessly with template systems."""

    def test_llm_request_serialization_for_templates(self):
        """Test that LLM requests can be serialized for template use."""
        request = LLMRequest(
            model="gpt-4.1-nano",
            prompt="Test prompt with {{variable}}",
            temperature=0.7,
            system_message="System with {{role}}",
            extra_params={"custom": "{{custom_value}}"},
        )

        # Test serialization to dict (for template engines)
        serialized = request.model_dump()

        assert serialized["model"] == "gpt-4.1-nano"
        assert serialized["prompt"] == "Test prompt with {{variable}}"
        assert serialized["system_message"] == "System with {{role}}"
        assert serialized["extra_params"]["custom"] == "{{custom_value}}"

        # Test that template variables are preserved
        assert "{{variable}}" in serialized["prompt"]
        assert "{{role}}" in serialized["system_message"]
        assert "{{custom_value}}" in serialized["extra_params"]["custom"]

    def test_llm_request_from_template_dict(self):
        """Test creating LLM request from template-processed dict."""
        # Simulated template output (after variable substitution)
        template_output = {
            "model": "claude-3-5-haiku-20241022",
            "prompt": "Generate a physics question about waves",
            "temperature": 0.6,
            "max_tokens": 1500,
            "system_message": "You are a physics teacher",
            "extra_params": {"top_k": 40, "thinking": {"budget_tokens": 512}},
        }

        # Create LLM request from template dict
        request = LLMRequest(**template_output)

        assert request.model == "claude-3-5-haiku-20241022"
        assert request.prompt == "Generate a physics question about waves"
        assert request.temperature == 0.6
        assert request.max_tokens == 1500
        assert request.system_message == "You are a physics teacher"
        assert request.extra_params["top_k"] == 40
        assert request.extra_params["thinking"]["budget_tokens"] == 512

    def test_template_variable_validation_in_requests(self):
        """Test that template variables don't break request validation."""
        # Request with unresolved template variables
        request_with_templates = LLMRequest(
            model="{{model_name}}",  # Template variable
            prompt="Generate {{content_type}} about {{topic}}",
            temperature=0.7,
            system_message="You are a {{role}} specializing in {{subject}}",
        )

        # Should still be valid (validation happens after template processing)
        assert request_with_templates.model == "{{model_name}}"
        assert "{{content_type}}" in request_with_templates.prompt
        assert "{{topic}}" in request_with_templates.prompt
        assert "{{role}}" in request_with_templates.system_message
        assert "{{subject}}" in request_with_templates.system_message

        # After template processing, should create valid resolved request
        resolved_data = {
            "model": "gpt-4.1-nano",
            "prompt": "Generate questions about algebra",
            "temperature": 0.7,
            "system_message": "You are a teacher specializing in mathematics",
        }

        resolved_request = LLMRequest(**resolved_data)
        assert resolved_request.model == "gpt-4.1-nano"
        assert resolved_request.prompt == "Generate questions about algebra"
        assert resolved_request.system_message == "You are a teacher specializing in mathematics"
