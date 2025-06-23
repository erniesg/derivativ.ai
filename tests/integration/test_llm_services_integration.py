"""
Integration tests for LLM services with configuration loading.
Tests real configuration loading with mocked API calls for all providers.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.config import get_config_manager
from src.models.llm_models import LLMRequest
from src.services.anthropic import AnthropicLLMService
from src.services.gemini import GeminiLLMService
from src.services.llm_factory import LLMFactory, LLMRouter
from src.services.openai import OpenAILLMService

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestOpenAIServiceIntegration:
    """Integration tests for OpenAI service with real config."""

    @pytest.fixture
    def config_manager(self):
        """Get config manager with test configuration."""
        return get_config_manager()

    @pytest.fixture
    def openai_config(self, config_manager):
        """Get OpenAI provider configuration."""
        app_config = config_manager.load_config()
        return app_config.llm_providers.openai

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-openai-key"})
    @patch("src.services.openai.AsyncOpenAI")
    def test_service_creation_from_config(self, mock_openai_client, openai_config):
        """Test creating OpenAI service from configuration."""
        service = OpenAILLMService.from_config(openai_config)

        assert service.config["default_model"] == openai_config.default_model
        assert service.config["timeout_seconds"] == openai_config.timeout_seconds
        assert service.config["max_retries"] == openai_config.max_retries

        # Verify client was initialized with correct parameters
        mock_openai_client.assert_called_once()
        call_kwargs = mock_openai_client.call_args.kwargs
        assert call_kwargs["api_key"] == "test-openai-key"
        assert call_kwargs["timeout"] == openai_config.timeout_seconds
        assert call_kwargs["max_retries"] == openai_config.max_retries

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-openai-key"})
    @patch("src.services.openai.AsyncOpenAI")
    def test_service_creation_missing_config_value(self, mock_openai_client):
        """Test service creation when config value is missing."""
        from src.models.llm_models import LLMProviderConfig

        # Create config with missing API key env
        config = LLMProviderConfig(
            default_model="gpt-4o-mini",
            api_key_env="MISSING_API_KEY",  # This env var doesn't exist
            timeout_seconds=30,
            max_retries=3,
        )

        with pytest.raises(ValueError, match="API key not found"):
            OpenAILLMService.from_config(config)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-openai-key"})
    @patch("src.services.openai.AsyncOpenAI")
    async def test_generate_with_config_defaults(self, mock_openai_client, openai_config):
        """Test generation using configuration defaults."""
        # Mock the OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response from OpenAI"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 15
        mock_response.usage.total_tokens = 25

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai_client.return_value = mock_client_instance

        service = OpenAILLMService.from_config(openai_config)

        # Use config default model
        request = LLMRequest(model=openai_config.default_model, prompt="Test prompt", stream=False)

        response = await service.generate_non_stream(request)

        assert response.content == "Test response from OpenAI"
        assert response.model_used == openai_config.default_model
        assert response.provider == "openai"
        assert response.tokens_used == 25

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-openai-key"})
    @patch("src.services.openai.AsyncOpenAI")
    async def test_streaming_with_config(self, mock_openai_client, openai_config):
        """Test streaming generation with configuration."""

        # Mock streaming response
        async def mock_stream():
            mock_chunk1 = Mock()
            mock_chunk1.choices = [Mock()]
            mock_chunk1.choices[0].delta.content = "Streaming "
            mock_chunk1.usage = None

            mock_chunk2 = Mock()
            mock_chunk2.choices = [Mock()]
            mock_chunk2.choices[0].delta.content = "response"
            mock_chunk2.usage = None

            # Final chunk with usage
            mock_final = Mock()
            mock_final.choices = [Mock()]
            mock_final.choices[0].delta.content = None
            mock_final.usage = Mock()
            mock_final.usage.prompt_tokens = 8
            mock_final.usage.completion_tokens = 12
            mock_final.usage.total_tokens = 20

            for chunk in [mock_chunk1, mock_chunk2, mock_final]:
                yield chunk

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_stream())
        mock_openai_client.return_value = mock_client_instance

        service = OpenAILLMService.from_config(openai_config)

        request = LLMRequest(
            model=openai_config.default_model, prompt="Test streaming", stream=True
        )

        chunks = []
        async for chunk in service.generate_stream(request):
            chunks.append(chunk)

        # Should have content chunks + done chunk
        content_chunks = [c for c in chunks if c.chunk_type.value == "content"]
        done_chunks = [c for c in chunks if c.chunk_type.value == "done"]

        assert len(content_chunks) >= 2
        assert len(done_chunks) == 1

        # Verify final accumulated content
        total_content = "".join(c.delta for c in content_chunks)
        assert "Streaming response" in total_content


class TestAnthropicServiceIntegration:
    """Integration tests for Anthropic service with real config."""

    @pytest.fixture
    def config_manager(self):
        """Get config manager with test configuration."""
        return get_config_manager()

    @pytest.fixture
    def anthropic_config(self, config_manager):
        """Get Anthropic provider configuration."""
        app_config = config_manager.load_config()
        return app_config.llm_providers.anthropic

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-anthropic-key"})
    @patch("src.services.anthropic.AsyncAnthropic")
    def test_service_creation_from_config(self, mock_anthropic_client, anthropic_config):
        """Test creating Anthropic service from configuration."""
        service = AnthropicLLMService.from_config(anthropic_config)

        assert service.config["default_model"] == anthropic_config.default_model
        assert service.config["timeout_seconds"] == anthropic_config.timeout_seconds
        assert service.config["max_retries"] == anthropic_config.max_retries

        # Verify client was initialized with correct parameters
        mock_anthropic_client.assert_called_once()
        call_kwargs = mock_anthropic_client.call_args.kwargs
        assert call_kwargs["api_key"] == "test-anthropic-key"
        assert call_kwargs["timeout"] == anthropic_config.timeout_seconds
        assert call_kwargs["max_retries"] == anthropic_config.max_retries

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-anthropic-key"})
    @patch("src.services.anthropic.AsyncAnthropic")
    async def test_generate_with_system_message(self, mock_anthropic_client, anthropic_config):
        """Test generation with system message (Anthropic-specific feature)."""
        # Mock the Anthropic response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response from Claude"
        mock_response.usage.input_tokens = 12
        mock_response.usage.output_tokens = 18
        mock_response.stop_reason = "end_turn"

        mock_client_instance = Mock()
        mock_client_instance.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic_client.return_value = mock_client_instance

        service = AnthropicLLMService.from_config(anthropic_config)

        request = LLMRequest(
            model=anthropic_config.default_model,
            prompt="What is AI?",
            system_message="You are a helpful AI assistant.",
            stream=False,
        )

        response = await service.generate_non_stream(request)

        assert response.content == "Test response from Claude"
        assert response.model_used == anthropic_config.default_model
        assert response.provider == "anthropic"

        # Verify system message was passed correctly
        create_call = mock_client_instance.messages.create.call_args
        assert create_call.kwargs["system"] == "You are a helpful AI assistant."
        assert create_call.kwargs["messages"][0]["content"] == "What is AI?"


class TestGeminiServiceIntegration:
    """Integration tests for Google Gemini service with real config."""

    @pytest.fixture
    def config_manager(self):
        """Get config manager with test configuration."""
        return get_config_manager()

    @pytest.fixture
    def gemini_config(self, config_manager):
        """Get Google provider configuration."""
        app_config = config_manager.load_config()
        return app_config.llm_providers.google

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-google-key"})
    @patch("src.services.gemini.genai.configure")
    @patch("src.services.gemini.genai.GenerativeModel")
    def test_service_creation_from_config(self, mock_model, mock_configure, gemini_config):
        """Test creating Gemini service from configuration."""
        service = GeminiLLMService.from_config(gemini_config)

        assert service.config["default_model"] == gemini_config.default_model
        assert service.config["timeout_seconds"] == gemini_config.timeout_seconds
        assert service.config["max_retries"] == gemini_config.max_retries

        # Verify API was configured
        mock_configure.assert_called_once_with(api_key="test-google-key")

        # Verify model was initialized
        mock_model.assert_called_once()
        model_init_kwargs = mock_model.call_args.kwargs
        assert model_init_kwargs["model_name"] == gemini_config.default_model

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-google-key"})
    @patch("src.services.gemini.genai.configure")
    @patch("src.services.gemini.genai.GenerativeModel")
    async def test_generate_with_safety_settings(
        self, mock_model_class, mock_configure, gemini_config
    ):
        """Test generation with safety settings (Gemini-specific feature)."""
        # Mock the Gemini response
        mock_response = Mock()
        mock_response.text = "Test response from Gemini"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 14
        mock_response.safety_ratings = []

        mock_model_instance = Mock()
        mock_model_instance.generate_content_async = AsyncMock(return_value=mock_response)
        mock_model_class.return_value = mock_model_instance

        service = GeminiLLMService.from_config(gemini_config)

        request = LLMRequest(
            model=gemini_config.default_model,
            prompt="Tell me about space exploration",
            stream=False,
        )

        response = await service.generate_non_stream(request)

        assert response.content == "Test response from Gemini"
        assert response.model_used == gemini_config.default_model
        assert response.provider == "google"
        assert response.tokens_used == 24  # 10 + 14

        # Verify safety ratings are included in metadata
        assert "safety_ratings" in response.metadata

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-google-key"})
    @patch("src.services.gemini.genai.configure")
    @patch("src.services.gemini.genai.GenerativeModel")
    async def test_free_tier_cost_calculation(
        self, mock_model_class, mock_configure, gemini_config
    ):
        """Test that free tier models have zero cost."""
        # Ensure we're testing with the free tier model
        assert gemini_config.default_model == "gemini-2.0-flash-exp"

        mock_response = Mock()
        mock_response.text = "Free response"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 200

        mock_model_instance = Mock()
        mock_model_instance.generate_content_async = AsyncMock(return_value=mock_response)
        mock_model_class.return_value = mock_model_instance

        service = GeminiLLMService.from_config(gemini_config)

        request = LLMRequest(
            model=gemini_config.default_model,  # Free tier model
            prompt="Generate a lot of text",
            stream=False,
        )

        response = await service.generate_non_stream(request)

        # Even with many tokens, cost should be zero for free tier
        assert response.cost_estimate == 0.0


class TestLLMFactoryIntegration:
    """Integration tests for LLM Factory with real configuration."""

    @pytest.fixture
    def config_manager(self):
        """Get config manager with test configuration."""
        return get_config_manager()

    @pytest.fixture
    def factory_config(self, config_manager):
        """Create factory config from loaded configuration."""
        app_config = config_manager.load_config()

        return {
            "providers": {
                "openai": app_config.llm_providers.openai.model_dump(),
                "anthropic": app_config.llm_providers.anthropic.model_dump(),
                "google": app_config.llm_providers.google.model_dump(),
            }
        }

    @patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "GOOGLE_API_KEY": "test-google-key",
        },
    )
    def test_factory_with_real_config(self, factory_config):
        """Test factory initialization with real configuration."""
        factory = LLMFactory(config=factory_config)

        # Verify provider configs were loaded
        assert "openai" in factory._provider_configs
        assert "anthropic" in factory._provider_configs
        assert "google" in factory._provider_configs

        # Verify config values
        assert factory._provider_configs["openai"].default_model == "gpt-4o-mini"
        assert factory._provider_configs["anthropic"].default_model == "claude-3-5-haiku-20241022"
        assert factory._provider_configs["google"].default_model == "gemini-2.0-flash-exp"

    @patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "GOOGLE_API_KEY": "test-google-key",
        },
    )
    def test_router_with_real_config(self, factory_config):
        """Test router with real configuration."""
        router_config = {
            **factory_config,
            "enable_fallback": True,
            "fallback_providers": ["openai", "anthropic", "google"],
        }

        router = LLMRouter(config=router_config)

        # Test model detection
        assert router.factory.detect_provider("gpt-4o-mini") == "openai"
        assert router.factory.detect_provider("claude-3-5-haiku-20241022") == "anthropic"
        assert router.factory.detect_provider("gemini-2.0-flash-exp") == "google"

        # Test service creation (services are created and cached)
        openai_service = router.factory.get_service("openai")
        anthropic_service = router.factory.get_service("anthropic")
        gemini_service = router.factory.get_service("google")

        # Verify services were created correctly
        assert openai_service is not None
        assert anthropic_service is not None
        assert gemini_service is not None

        # Verify services are cached (same instance returned)
        assert router.factory.get_service("openai") is openai_service
        assert router.factory.get_service("anthropic") is anthropic_service
        assert router.factory.get_service("google") is gemini_service

    async def test_router_end_to_end_with_config(self, factory_config):
        """Test complete router workflow with configuration."""
        router_config = {
            **factory_config,
            "enable_fallback": True,
            "fallback_providers": ["openai", "anthropic", "google"],
        }

        router = LLMRouter(config=router_config)

        # Mock all services to avoid actual API calls
        with patch.object(router.factory, "get_service") as mock_get_service:
            mock_service = Mock()
            mock_response = Mock()
            mock_response.content = "Integration test response"
            mock_response.model_used = "gpt-4o-mini"
            mock_response.provider = "openai"
            mock_service.generate = AsyncMock(return_value=mock_response)
            mock_get_service.return_value = mock_service

            request = LLMRequest(
                model="gpt-4o-mini", prompt="Integration test prompt", stream=False
            )

            response = await router.generate(request)

            assert response.content == "Integration test response"
            assert response.model_used == "gpt-4o-mini"
            assert response.provider == "openai"


class TestConfigurationValidation:
    """Test configuration validation and error handling."""

    def test_invalid_provider_config(self):
        """Test handling of invalid provider configuration."""
        invalid_config = {
            "providers": {
                "openai": {
                    "default_model": "gpt-4o-mini",
                    "api_key_env": "OPENAI_API_KEY",
                    "timeout_seconds": -1,  # Invalid: negative timeout
                    "max_retries": 3,
                }
            }
        }

        # Factory should handle invalid config gracefully
        factory = LLMFactory(config=invalid_config)

        # Should not have loaded the invalid openai config
        assert "openai" not in factory._provider_configs

    @patch.dict("os.environ", {}, clear=True)  # Clear all env vars
    def test_missing_api_keys(self):
        """Test behavior when API keys are missing."""
        factory = LLMFactory()

        # Should raise errors for missing API keys
        with pytest.raises(ValueError, match="OpenAI API key not found"):
            factory.get_service("openai")

        with pytest.raises(ValueError, match="Anthropic API key not found"):
            factory.get_service("anthropic")

        with pytest.raises(ValueError, match="Google API key not found"):
            factory.get_service("google")

    def test_partial_configuration(self):
        """Test factory with partial provider configuration."""
        partial_config = {
            "providers": {
                "openai": {
                    "default_model": "gpt-4o-mini",
                    "api_key_env": "OPENAI_API_KEY",
                    "timeout_seconds": 30,
                    "max_retries": 3,
                }
                # Missing anthropic and google configs
            }
        }

        factory = LLMFactory(config=partial_config)

        # Should have openai config but not others
        assert "openai" in factory._provider_configs
        assert "anthropic" not in factory._provider_configs
        assert "google" not in factory._provider_configs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
