"""
Integration tests for configuration system with LLM models.
Tests written first following TDD approach.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# Import models and config system we're about to create
from src.core.config import ConfigManager
from src.models.llm_models import (
    AppConfig,
    LLMDefaults,
    LLMProvidersConfig,
    LLMRequest,
    LLMResponse,
)


class TestConfigManagerIntegration:
    """Test integration between configuration management and LLM models."""

    @pytest.fixture
    def sample_config_yaml(self):
        """Sample YAML configuration for testing."""
        return """
# LLM Provider Configuration
llm_providers:
  openai:
    default_model: "gpt-4.1-nano"
    api_key_env: "OPENAI_API_KEY"
    base_url: "https://api.openai.com/v1"

  anthropic:
    default_model: "claude-3-5-haiku-20241022"
    api_key_env: "ANTHROPIC_API_KEY"
    base_url: "https://api.anthropic.com"

  google:
    default_model: "gemini-2.5-flash-lite-preview-06-17"
    api_key_env: "GOOGLE_API_KEY"

  default_provider: "openai"

# LLM Defaults
llm_defaults:
  temperature: 0.7
  max_tokens: 1000
  top_p: 0.9
  timeout_seconds: 30

# Quality Control (existing)
quality_control:
  thresholds:
    auto_approve: 0.85
    manual_review: 0.70
    refine: 0.60
    regenerate: 0.40
    reject: 0.20

# Agent Configuration (updated)
agents:
  question_generator:
    model: "gpt-4.1-nano"
    temperature: 0.8
    max_tokens: 2000
    extra_params:
      frequency_penalty: 0.1
      presence_penalty: 0.0

  marker:
    model: "claude-3-5-haiku-20241022"
    temperature: 0.3
    max_tokens: 1500
    extra_params:
      top_k: 40

  reviewer:
    model: "claude-3-5-haiku-20241022"
    temperature: 0.5
    max_tokens: 1000
    extra_params:
      top_k: 20

  refinement:
    model: "gpt-4.1-nano"
    temperature: 0.7
    max_tokens: 2500
    extra_params:
      frequency_penalty: 0.2
"""

    @pytest.fixture
    def temp_config_file(self, sample_config_yaml):
        """Create temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(sample_config_yaml)
            temp_path = f.name

        yield temp_path

        # Cleanup
        os.unlink(temp_path)

    @pytest.fixture
    def sample_env_vars(self):
        """Sample environment variables for testing."""
        return {
            "OPENAI_API_KEY": "sk-test-openai-key",
            "ANTHROPIC_API_KEY": "sk-ant-test-key",
            "GOOGLE_API_KEY": "test-google-key",
            "LLM_DEFAULT_TEMPERATURE": "0.8",  # Override default
            "LLM_DEFAULT_MAX_TOKENS": "1200",  # Override default
        }

    def test_load_config_from_yaml(self, temp_config_file):
        """Test loading configuration from YAML file."""
        config_manager = ConfigManager(config_path=temp_config_file)
        config = config_manager.load_config()

        # Test LLM providers loaded correctly
        assert isinstance(config, AppConfig)
        assert isinstance(config.llm_providers, LLMProvidersConfig)

        # Test OpenAI config
        assert config.llm_providers.openai.default_model == "gpt-4.1-nano"
        assert config.llm_providers.openai.api_key_env == "OPENAI_API_KEY"
        assert config.llm_providers.openai.base_url == "https://api.openai.com/v1"

        # Test Anthropic config
        assert config.llm_providers.anthropic.default_model == "claude-3-5-haiku-20241022"
        assert config.llm_providers.anthropic.api_key_env == "ANTHROPIC_API_KEY"

        # Test Google config
        assert config.llm_providers.google.default_model == "gemini-2.5-flash-lite-preview-06-17"
        assert config.llm_providers.google.api_key_env == "GOOGLE_API_KEY"

        # Test default provider
        assert config.llm_providers.default_provider == "openai"

        # Test LLM defaults
        assert isinstance(config.llm_defaults, LLMDefaults)
        assert config.llm_defaults.temperature == 0.7
        assert config.llm_defaults.max_tokens == 1000
        assert config.llm_defaults.top_p == 0.9
        assert config.llm_defaults.timeout_seconds == 30

        # Test agent configurations
        assert config.agents["question_generator"]["model"] == "gpt-4.1-nano"
        assert config.agents["question_generator"]["temperature"] == 0.8
        assert config.agents["question_generator"]["extra_params"]["frequency_penalty"] == 0.1

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "sk-test-openai",
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "GOOGLE_API_KEY": "test-google",
            "LLM_DEFAULT_TEMPERATURE": "0.9",
            "LLM_DEFAULT_MAX_TOKENS": "1500",
        },
    )
    def test_env_var_overrides(self, temp_config_file):
        """Test that environment variables override YAML config."""
        config_manager = ConfigManager(config_path=temp_config_file)
        config = config_manager.load_config()

        # Test environment variable overrides for defaults
        assert config.llm_defaults.temperature == 0.9  # Overridden from env
        assert config.llm_defaults.max_tokens == 1500  # Overridden from env
        assert config.llm_defaults.top_p == 0.9  # From YAML (not overridden)

    def test_create_llm_request_from_config(self, temp_config_file):
        """Test creating LLM request using configuration defaults."""
        config_manager = ConfigManager(config_path=temp_config_file)
        config = config_manager.load_config()

        # Create request using config defaults
        request = LLMRequest(
            model=config.llm_providers.openai.default_model,
            prompt="Test prompt",
            temperature=config.llm_defaults.temperature,
            max_tokens=config.llm_defaults.max_tokens,
            top_p=config.llm_defaults.top_p,
        )

        assert request.model == "gpt-4.1-nano"
        assert request.temperature == 0.7
        assert request.max_tokens == 1000
        assert request.top_p == 0.9

    def test_create_agent_specific_llm_request(self, temp_config_file):
        """Test creating LLM request using agent-specific configuration."""
        config_manager = ConfigManager(config_path=temp_config_file)
        config = config_manager.load_config()

        # Get question generator agent config
        agent_config = config.agents["question_generator"]

        # Create request with agent-specific settings
        request = LLMRequest(
            model=agent_config["model"],
            prompt="Generate a question",
            temperature=agent_config["temperature"],
            max_tokens=agent_config["max_tokens"],
            extra_params=agent_config["extra_params"],
        )

        assert request.model == "gpt-4.1-nano"
        assert request.temperature == 0.8  # Agent override
        assert request.max_tokens == 2000  # Agent override
        assert request.extra_params["frequency_penalty"] == 0.1
        assert request.extra_params["presence_penalty"] == 0.0

    def test_runtime_config_override(self, temp_config_file):
        """Test runtime configuration overrides."""
        config_manager = ConfigManager(config_path=temp_config_file)
        config = config_manager.load_config()

        # Base request from config
        base_request = LLMRequest(
            model=config.llm_providers.openai.default_model,
            prompt="Test prompt",
            temperature=config.llm_defaults.temperature,
            max_tokens=config.llm_defaults.max_tokens,
        )

        # Runtime overrides
        runtime_overrides = {
            "model": "claude-3-5-haiku-20241022",  # Switch provider
            "temperature": 0.2,  # Override temperature
            "extra_params": {"top_k": 40},  # Add provider-specific param
        }

        # Create overridden request
        overridden_request = LLMRequest(**{**base_request.model_dump(), **runtime_overrides})

        assert overridden_request.model == "claude-3-5-haiku-20241022"  # Overridden
        assert overridden_request.temperature == 0.2  # Overridden
        assert overridden_request.max_tokens == 1000  # From config
        assert overridden_request.extra_params["top_k"] == 40  # Added

    def test_provider_fallback_configuration(self, temp_config_file):
        """Test provider fallback using configuration."""
        config_manager = ConfigManager(config_path=temp_config_file)
        config = config_manager.load_config()

        # Test primary provider
        primary_model = config.llm_providers.openai.default_model
        primary_request = LLMRequest(model=primary_model, prompt="Test with primary provider")
        assert primary_request.model == "gpt-4.1-nano"

        # Test fallback to anthropic
        fallback_model = config.llm_providers.anthropic.default_model
        fallback_request = LLMRequest(model=fallback_model, prompt="Test with fallback provider")
        assert fallback_request.model == "claude-3-5-haiku-20241022"

    def test_config_validation_with_invalid_values(self, temp_config_file):
        """Test that configuration validation catches invalid values."""
        # Load valid config first
        config_manager = ConfigManager(config_path=temp_config_file)
        config = config_manager.load_config()

        # Test creating invalid request using config (should fail validation)
        with pytest.raises(ValueError):
            LLMRequest(
                model=config.llm_providers.openai.default_model,
                prompt="Test prompt",
                temperature=3.0,  # Invalid temperature from hypothetical bad config
                max_tokens=config.llm_defaults.max_tokens,
            )

    def test_missing_config_file_defaults(self):
        """Test that missing config file uses sensible defaults."""
        # Test with non-existent config file
        config_manager = ConfigManager(config_path="/non/existent/file.yaml")
        config = config_manager.load_config()  # Should use defaults

        # Should create valid config with defaults
        assert isinstance(config, AppConfig)
        assert isinstance(config.llm_providers, LLMProvidersConfig)
        assert isinstance(config.llm_defaults, LLMDefaults)

        # Test default values
        assert config.llm_providers.default_provider in ["openai", "anthropic", "google"]
        assert 0.0 <= config.llm_defaults.temperature <= 2.0
        assert config.llm_defaults.max_tokens > 0

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "sk-test-key",
            "LLM_PROVIDERS_DEFAULT_PROVIDER": "anthropic",  # Override default provider
        },
    )
    def test_env_override_provider_selection(self, temp_config_file):
        """Test environment variable override of default provider."""
        config_manager = ConfigManager(config_path=temp_config_file)
        config = config_manager.load_config()

        # Default provider should be overridden by env var
        assert config.llm_providers.default_provider == "anthropic"

        # Create request using overridden default provider
        default_model = getattr(
            config.llm_providers, config.llm_providers.default_provider
        ).default_model

        request = LLMRequest(model=default_model, prompt="Test with env-overridden provider")

        assert request.model == "claude-3-5-haiku-20241022"  # Anthropic model


class TestConfigLLMServiceIntegration:
    """Test integration between config system and LLM service implementations."""

    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service for testing."""
        service = MagicMock()
        service.generate.return_value = LLMResponse(
            content="Generated content",
            model_used="gpt-4.1-nano",
            tokens_used=50,
            cost_estimate=0.000025,
            latency_ms=1000,
            provider="openai",
        )
        return service

    def test_service_uses_config_defaults(self, temp_config_file, mock_llm_service):
        """Test that LLM service uses configuration defaults."""
        # This test demonstrates how services would integrate with config
        config_manager = ConfigManager(config_path=temp_config_file)
        config = config_manager.load_config()

        # Service would be initialized with config
        # service = OpenAILLMService(config=config.llm_providers.openai)

        # Create request using config defaults
        request = LLMRequest(
            model=config.llm_providers.openai.default_model,
            prompt="Test prompt",
            temperature=config.llm_defaults.temperature,
            max_tokens=config.llm_defaults.max_tokens,
        )

        # Mock service call
        response = mock_llm_service.generate(request)

        # Verify service was called with config values
        mock_llm_service.generate.assert_called_once_with(request)
        assert response.model_used == "gpt-4.1-nano"
        assert response.provider == "openai"

    def test_agent_uses_config_specific_settings(self, temp_config_file, mock_llm_service):
        """Test that agents use their specific configuration settings."""
        config_manager = ConfigManager(config_path=temp_config_file)
        config = config_manager.load_config()

        # Get marker agent config (uses Anthropic)
        marker_config = config.agents["marker"]

        # Create request with marker-specific settings
        request = LLMRequest(
            model=marker_config["model"],
            prompt="Create marking scheme",
            temperature=marker_config["temperature"],
            max_tokens=marker_config["max_tokens"],
            extra_params=marker_config["extra_params"],
        )

        # Mock anthropic service response
        mock_llm_service.generate.return_value = LLMResponse(
            content="Marking scheme generated",
            model_used="claude-3-5-haiku-20241022",
            tokens_used=75,
            cost_estimate=0.00006,
            latency_ms=1200,
            provider="anthropic",
        )

        response = mock_llm_service.generate(request)

        # Verify agent-specific configuration was used
        assert request.model == "claude-3-5-haiku-20241022"
        assert request.temperature == 0.3  # Marker-specific
        assert request.max_tokens == 1500  # Marker-specific
        assert request.extra_params["top_k"] == 40  # Marker-specific
        assert response.provider == "anthropic"

    def test_config_hot_reload_affects_new_requests(self, temp_config_file):
        """Test that configuration changes affect new requests."""
        config_manager = ConfigManager(config_path=temp_config_file)
        config = config_manager.load_config()

        # Create initial request
        initial_request = LLMRequest(
            model=config.llm_providers.openai.default_model,
            prompt="Initial prompt",
            temperature=config.llm_defaults.temperature,
        )
        assert initial_request.temperature == 0.7

        # Simulate config change (in real implementation, this would reload config)
        updated_config_data = config.model_dump()
        updated_config_data["llm_defaults"]["temperature"] = 0.9
        updated_config = AppConfig(**updated_config_data)

        # Create new request with updated config
        updated_request = LLMRequest(
            model=updated_config.llm_providers.openai.default_model,
            prompt="Updated prompt",
            temperature=updated_config.llm_defaults.temperature,
        )
        assert updated_request.temperature == 0.9
