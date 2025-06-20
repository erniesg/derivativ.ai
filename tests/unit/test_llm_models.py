"""
Unit tests for LLM Pydantic models and configuration.
Tests written first following TDD approach.
"""


import pytest
from pydantic import ValidationError

# Import the models we're about to create
from src.models.llm_models import (
    AppConfig,
    LLMDefaults,
    LLMProviderConfig,
    LLMProvidersConfig,
    LLMRequest,
    LLMResponse,
)


class TestLLMRequest:
    """Test LLM request model validation and defaults."""

    def test_minimal_valid_request(self):
        """Test creating a minimal valid LLM request."""
        request = LLMRequest(model="gpt-4.1-nano", prompt="Hello world")

        assert request.model == "gpt-4.1-nano"
        assert request.prompt == "Hello world"
        assert request.temperature == 0.7  # default
        assert request.max_tokens == 1000  # default
        assert request.top_p == 1.0  # default
        assert request.system_message is None  # default
        assert request.stop_sequences == []  # default
        assert request.extra_params == {}  # default

    def test_request_with_all_fields(self):
        """Test creating request with all fields specified."""
        request = LLMRequest(
            model="claude-3-5-haiku-20241022",
            prompt="Generate a math question",
            temperature=0.8,
            max_tokens=1500,
            top_p=0.9,
            system_message="You are a math tutor",
            stop_sequences=["END", "STOP"],
            extra_params={"top_k": 40, "thinking": {"budget_tokens": 1024}},
        )

        assert request.model == "claude-3-5-haiku-20241022"
        assert request.prompt == "Generate a math question"
        assert request.temperature == 0.8
        assert request.max_tokens == 1500
        assert request.top_p == 0.9
        assert request.system_message == "You are a math tutor"
        assert request.stop_sequences == ["END", "STOP"]
        assert request.extra_params["top_k"] == 40
        assert request.extra_params["thinking"]["budget_tokens"] == 1024

    def test_invalid_temperature_range(self):
        """Test validation fails for invalid temperature values."""
        with pytest.raises(ValidationError) as exc_info:
            LLMRequest(
                model="gpt-4.1-nano",
                prompt="test",
                temperature=3.0,  # too high
            )
        assert "less than or equal to 2" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            LLMRequest(
                model="gpt-4.1-nano",
                prompt="test",
                temperature=-0.1,  # too low
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_invalid_top_p_range(self):
        """Test validation fails for invalid top_p values."""
        with pytest.raises(ValidationError) as exc_info:
            LLMRequest(
                model="gpt-4.1-nano",
                prompt="test",
                top_p=1.5,  # too high
            )
        assert "less than or equal to 1" in str(exc_info.value)

    def test_invalid_max_tokens(self):
        """Test validation fails for invalid max_tokens values."""
        with pytest.raises(ValidationError) as exc_info:
            LLMRequest(
                model="gpt-4.1-nano",
                prompt="test",
                max_tokens=0,  # too low
            )
        assert "greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            LLMRequest(
                model="gpt-4.1-nano",
                prompt="test",
                max_tokens=1000000,  # too high
            )
        assert "less than or equal to 100000" in str(exc_info.value)

    def test_empty_prompt_validation(self):
        """Test validation fails for empty prompts."""
        with pytest.raises(ValidationError) as exc_info:
            LLMRequest(
                model="gpt-4.1-nano",
                prompt="",  # empty
            )
        assert "prompt cannot be empty" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            LLMRequest(
                model="gpt-4.1-nano",
                prompt="   ",  # whitespace only
            )
        assert "prompt cannot be empty" in str(exc_info.value)

    def test_model_name_validation(self):
        """Test validation fails for invalid model names."""
        with pytest.raises(ValidationError) as exc_info:
            LLMRequest(
                model="",  # empty
                prompt="test",
            )
        assert "model cannot be empty" in str(exc_info.value)


class TestLLMResponse:
    """Test LLM response model validation."""

    def test_valid_response(self):
        """Test creating a valid LLM response."""
        response = LLMResponse(
            content="Here is a math question: What is 2+2?",
            model_used="gpt-4.1-nano",
            tokens_used=150,
            cost_estimate=0.000045,
            latency_ms=1200,
            provider="openai",
        )

        assert response.content == "Here is a math question: What is 2+2?"
        assert response.model_used == "gpt-4.1-nano"
        assert response.tokens_used == 150
        assert response.cost_estimate == 0.000045
        assert response.latency_ms == 1200
        assert response.provider == "openai"

    def test_response_validation_negative_values(self):
        """Test validation fails for negative values."""
        with pytest.raises(ValidationError) as exc_info:
            LLMResponse(
                content="test",
                model_used="gpt-4.1-nano",
                tokens_used=-1,  # negative
                cost_estimate=0.0001,
                latency_ms=1000,
                provider="openai",
            )
        assert "greater than or equal to 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            LLMResponse(
                content="test",
                model_used="gpt-4.1-nano",
                tokens_used=100,
                cost_estimate=-0.0001,  # negative
                latency_ms=1000,
                provider="openai",
            )
        assert "greater than or equal to 0" in str(exc_info.value)


class TestLLMProviderConfig:
    """Test individual provider configuration model."""

    def test_minimal_provider_config(self):
        """Test minimal provider configuration."""
        config = LLMProviderConfig(default_model="gpt-4.1-nano", api_key_env="OPENAI_API_KEY")

        assert config.default_model == "gpt-4.1-nano"
        assert config.api_key_env == "OPENAI_API_KEY"
        assert config.base_url is None  # optional
        assert config.api_version is None  # optional

    def test_full_provider_config(self):
        """Test full provider configuration."""
        config = LLMProviderConfig(
            default_model="claude-3-5-haiku-20241022",
            api_key_env="ANTHROPIC_API_KEY",
            base_url="https://api.anthropic.com",
            api_version="2023-06-01",
        )

        assert config.default_model == "claude-3-5-haiku-20241022"
        assert config.api_key_env == "ANTHROPIC_API_KEY"
        assert config.base_url == "https://api.anthropic.com"
        assert config.api_version == "2023-06-01"


class TestLLMProvidersConfig:
    """Test combined providers configuration."""

    def test_valid_providers_config(self):
        """Test creating valid providers configuration."""
        config = LLMProvidersConfig(
            openai=LLMProviderConfig(
                default_model="gpt-4.1-nano",
                api_key_env="OPENAI_API_KEY",
                base_url="https://api.openai.com/v1",
            ),
            anthropic=LLMProviderConfig(
                default_model="claude-3-5-haiku-20241022",
                api_key_env="ANTHROPIC_API_KEY",
                base_url="https://api.anthropic.com",
            ),
            google=LLMProviderConfig(
                default_model="gemini-2.5-flash-lite-preview-06-17", api_key_env="GOOGLE_API_KEY"
            ),
            default_provider="openai",
        )

        assert config.openai.default_model == "gpt-4.1-nano"
        assert config.anthropic.default_model == "claude-3-5-haiku-20241022"
        assert config.google.default_model == "gemini-2.5-flash-lite-preview-06-17"
        assert config.default_provider == "openai"

    def test_invalid_default_provider(self):
        """Test validation fails for invalid default provider."""
        with pytest.raises(ValidationError) as exc_info:
            LLMProvidersConfig(
                openai=LLMProviderConfig(
                    default_model="gpt-4.1-nano", api_key_env="OPENAI_API_KEY"
                ),
                anthropic=LLMProviderConfig(
                    default_model="claude-3-5-haiku-20241022", api_key_env="ANTHROPIC_API_KEY"
                ),
                google=LLMProviderConfig(
                    default_model="gemini-2.5-flash-lite-preview-06-17",
                    api_key_env="GOOGLE_API_KEY",
                ),
                default_provider="invalid_provider",  # not in valid providers
            )
        assert "Input should be 'openai', 'anthropic' or 'google'" in str(exc_info.value)


class TestLLMDefaults:
    """Test LLM defaults configuration."""

    def test_valid_defaults(self):
        """Test creating valid LLM defaults."""
        defaults = LLMDefaults(temperature=0.7, max_tokens=1000, top_p=0.9, timeout_seconds=30)

        assert defaults.temperature == 0.7
        assert defaults.max_tokens == 1000
        assert defaults.top_p == 0.9
        assert defaults.timeout_seconds == 30

    def test_defaults_validation(self):
        """Test validation for defaults."""
        with pytest.raises(ValidationError) as exc_info:
            LLMDefaults(
                temperature=3.0,  # invalid
                max_tokens=1000,
                top_p=0.9,
                timeout_seconds=30,
            )
        assert "less than or equal to 2" in str(exc_info.value)


class TestAppConfig:
    """Test complete application configuration."""

    def test_valid_app_config(self):
        """Test creating valid application configuration."""
        config = AppConfig(
            llm_providers=LLMProvidersConfig(
                openai=LLMProviderConfig(
                    default_model="gpt-4.1-nano", api_key_env="OPENAI_API_KEY"
                ),
                anthropic=LLMProviderConfig(
                    default_model="claude-3-5-haiku-20241022", api_key_env="ANTHROPIC_API_KEY"
                ),
                google=LLMProviderConfig(
                    default_model="gemini-2.5-flash-lite-preview-06-17",
                    api_key_env="GOOGLE_API_KEY",
                ),
                default_provider="openai",
            ),
            llm_defaults=LLMDefaults(
                temperature=0.7, max_tokens=1000, top_p=0.9, timeout_seconds=30
            ),
            quality_control={
                "thresholds": {
                    "auto_approve": 0.85,
                    "manual_review": 0.70,
                    "refine": 0.60,
                    "regenerate": 0.40,
                    "reject": 0.20,
                }
            },
            agents={
                "question_generator": {
                    "model": "gpt-4.1-nano",
                    "temperature": 0.8,
                    "extra_params": {"frequency_penalty": 0.1},
                }
            },
        )

        assert config.llm_providers.default_provider == "openai"
        assert config.llm_defaults.temperature == 0.7
        assert config.quality_control["thresholds"]["auto_approve"] == 0.85
        assert config.agents["question_generator"]["model"] == "gpt-4.1-nano"


class TestJinjaTemplateIntegration:
    """Test integration between Pydantic models and Jinja templates."""

    def test_llm_request_to_template_variables(self):
        """Test converting LLM request to template variables."""
        request = LLMRequest(
            model="gpt-4.1-nano",
            prompt="Generate a question about {{topic}} for grade {{grade}}",
            system_message="You are a {{role}}",
            extra_params={"difficulty": "medium"},
        )

        # Test that request can be converted to dict for template rendering
        template_vars = request.model_dump()

        assert template_vars["model"] == "gpt-4.1-nano"
        assert template_vars["prompt"] == "Generate a question about {{topic}} for grade {{grade}}"
        assert template_vars["system_message"] == "You are a {{role}}"
        assert template_vars["extra_params"]["difficulty"] == "medium"

    def test_template_variable_substitution_in_request(self):
        """Test that template variables can be processed in LLM requests."""
        # This test demonstrates how PromptManager would work with LLMRequest
        template_request = LLMRequest(
            model="gpt-4.1-nano",
            prompt="Create a {{question_type}} question about {{topic}}",
            system_message="You are a {{grade}} level {{subject}} teacher",
        )

        # Variables that would come from PromptManager
        variables = {
            "question_type": "multiple choice",
            "topic": "algebra",
            "grade": "9th",
            "subject": "mathematics",
        }

        # Test that request structure supports template substitution
        expected_prompt = "Create a multiple choice question about algebra"
        expected_system = "You are a 9th level mathematics teacher"

        # The actual substitution would happen in PromptManager
        # This test just validates the structure supports it
        assert "{{question_type}}" in template_request.prompt
        assert "{{topic}}" in template_request.prompt
        assert "{{grade}}" in template_request.system_message
        assert "{{subject}}" in template_request.system_message

    def test_nested_template_variables_in_extra_params(self):
        """Test template variables work in extra_params."""
        request = LLMRequest(
            model="claude-3-5-haiku-20241022",
            prompt="Generate content",
            extra_params={
                "thinking": {
                    "instructions": "Consider {{context}} when reasoning",
                    "budget_tokens": "{{thinking_budget}}",
                },
                "custom_setting": "{{custom_value}}",
            },
        )

        # Test that nested template variables are preserved
        assert "{{context}}" in request.extra_params["thinking"]["instructions"]
        assert "{{thinking_budget}}" in str(request.extra_params["thinking"]["budget_tokens"])
        assert "{{custom_value}}" in request.extra_params["custom_setting"]


class TestLLMRequestRuntimeOverrides:
    """Test runtime parameter override functionality."""

    def test_request_with_runtime_overrides(self):
        """Test how runtime overrides would work with LLM requests."""
        base_request = LLMRequest(
            model="gpt-4.1-nano",
            prompt="Generate a question",
            temperature=0.7,  # default
            max_tokens=1000,  # default
        )

        # Test creating overridden request (how services would handle overrides)
        override_params = {
            "model": "claude-3-5-haiku-20241022",
            "temperature": 0.2,
            "extra_params": {"top_k": 40},
        }

        # Create new request with overrides
        overridden_request = LLMRequest(**{**base_request.model_dump(), **override_params})

        assert overridden_request.model == "claude-3-5-haiku-20241022"  # overridden
        assert overridden_request.temperature == 0.2  # overridden
        assert overridden_request.max_tokens == 1000  # preserved
        assert overridden_request.extra_params["top_k"] == 40  # added

    def test_partial_override_preservation(self):
        """Test that partial overrides preserve other values."""
        original = LLMRequest(
            model="gpt-4.1-nano",
            prompt="Original prompt",
            temperature=0.7,
            system_message="Original system",
            extra_params={"original_param": "value"},
        )

        # Override only temperature and add extra param
        overrides = {
            "temperature": 0.2,
            "extra_params": {**original.extra_params, "new_param": "new_value"},
        }

        updated = LLMRequest(**{**original.model_dump(), **overrides})

        assert updated.model == "gpt-4.1-nano"  # preserved
        assert updated.prompt == "Original prompt"  # preserved
        assert updated.temperature == 0.2  # overridden
        assert updated.system_message == "Original system"  # preserved
        assert updated.extra_params["original_param"] == "value"  # preserved
        assert updated.extra_params["new_param"] == "new_value"  # added
