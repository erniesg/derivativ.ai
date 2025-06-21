"""
Unit tests for LLM Factory and Router.
Tests intelligent model routing and provider management without making actual API calls.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.models.llm_models import LLMRequest, LLMResponse
from src.services.llm_factory import LLMFactory, LLMRouter, create_llm_factory, create_llm_router


class TestLLMFactory:
    """Test LLM Factory functionality."""

    def test_detect_provider_exact_mappings(self):
        """Test provider detection for exact model mappings."""
        factory = LLMFactory()
        
        # OpenAI models
        assert factory.detect_provider("gpt-4o") == "openai"
        assert factory.detect_provider("gpt-4o-mini") == "openai"
        assert factory.detect_provider("gpt-3.5-turbo") == "openai"
        
        # Anthropic models
        assert factory.detect_provider("claude-3-5-sonnet-20241022") == "anthropic"
        assert factory.detect_provider("claude-3-5-haiku-20241022") == "anthropic"
        
        # Google models
        assert factory.detect_provider("gemini-2.0-flash-exp") == "google"
        assert factory.detect_provider("gemini-1.5-pro") == "google"

    def test_detect_provider_prefixes(self):
        """Test provider detection using model prefixes."""
        factory = LLMFactory()
        
        # OpenAI prefixes
        assert factory.detect_provider("gpt-4-turbo") == "openai"
        assert factory.detect_provider("text-davinci-003") == "openai"
        
        # Anthropic prefixes
        assert factory.detect_provider("claude-4-opus") == "anthropic"
        
        # Google prefixes
        assert factory.detect_provider("gemini-3.0-ultra") == "google"
        assert factory.detect_provider("palm-2-chat") == "google"

    def test_detect_provider_unknown_fallback(self):
        """Test fallback to OpenAI for unknown models."""
        factory = LLMFactory()
        
        assert factory.detect_provider("unknown-model") == "openai"
        assert factory.detect_provider("custom-llm-v1") == "openai"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-openai-key"})
    @patch("src.services.llm_factory.OpenAILLMService")
    def test_get_service_openai_creation(self, mock_openai_service):
        """Test OpenAI service creation."""
        factory = LLMFactory()
        
        # First call creates service
        service1 = factory.get_service("openai")
        mock_openai_service.assert_called_once_with(api_key="test-openai-key")
        
        # Second call returns cached service
        service2 = factory.get_service("openai")
        assert service1 == service2
        # Should not create new service
        assert mock_openai_service.call_count == 1

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-anthropic-key"})
    @patch("src.services.llm_factory.AnthropicLLMService")
    def test_get_service_anthropic_creation(self, mock_anthropic_service):
        """Test Anthropic service creation."""
        factory = LLMFactory()
        
        service = factory.get_service("anthropic")
        mock_anthropic_service.assert_called_once_with(api_key="test-anthropic-key")

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-google-key"})
    @patch("src.services.llm_factory.GeminiLLMService")
    def test_get_service_google_creation(self, mock_gemini_service):
        """Test Google Gemini service creation."""
        factory = LLMFactory()
        
        service = factory.get_service("google")
        mock_gemini_service.assert_called_once_with(api_key="test-google-key")

    def test_get_service_unsupported_provider(self):
        """Test error for unsupported provider."""
        factory = LLMFactory()
        
        with pytest.raises(ValueError, match="Unsupported provider: unsupported"):
            factory.get_service("unsupported")

    def test_get_service_missing_api_key(self):
        """Test error when API key is missing."""
        factory = LLMFactory()
        
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key not found"):
                factory.get_service("openai")

    @patch("src.services.llm_factory.LLMFactory.get_service")
    def test_get_all_available_models(self, mock_get_service):
        """Test getting all available models from all providers."""
        # Mock services
        mock_openai = Mock()
        mock_openai.get_available_models.return_value = ["gpt-4o", "gpt-4o-mini"]
        
        mock_anthropic = Mock()
        mock_anthropic.get_available_models.return_value = ["claude-3-5-haiku-20241022"]
        
        mock_google = Mock()
        mock_google.get_available_models.return_value = ["gemini-2.0-flash-exp"]
        
        def mock_service_getter(provider):
            if provider == "openai":
                return mock_openai
            elif provider == "anthropic":
                return mock_anthropic
            elif provider == "google":
                return mock_google
        
        mock_get_service.side_effect = mock_service_getter
        
        factory = LLMFactory()
        models = factory.get_all_available_models()
        
        assert models["openai"] == ["gpt-4o", "gpt-4o-mini"]
        assert models["anthropic"] == ["claude-3-5-haiku-20241022"]
        assert models["google"] == ["gemini-2.0-flash-exp"]

    @patch("src.services.llm_factory.LLMFactory.get_service")
    def test_get_all_available_models_with_failure(self, mock_get_service):
        """Test getting models when one provider fails."""
        def mock_service_getter(provider):
            if provider == "openai":
                mock_service = Mock()
                mock_service.get_available_models.return_value = ["gpt-4o"]
                return mock_service
            else:
                raise Exception(f"Provider {provider} unavailable")
        
        mock_get_service.side_effect = mock_service_getter
        
        factory = LLMFactory()
        models = factory.get_all_available_models()
        
        assert models["openai"] == ["gpt-4o"]
        assert models["anthropic"] == []
        assert models["google"] == []

    def test_factory_with_provider_configs(self):
        """Test factory initialization with provider configurations."""
        config = {
            "providers": {
                "openai": {
                    "default_model": "gpt-4o-mini",
                    "api_key_env": "OPENAI_API_KEY",
                    "timeout_seconds": 30,
                    "max_retries": 3
                }
            }
        }
        
        factory = LLMFactory(config=config)
        assert "openai" in factory._provider_configs
        assert factory._provider_configs["openai"].default_model == "gpt-4o-mini"


class TestLLMRouter:
    """Test LLM Router functionality."""

    @pytest.fixture
    def mock_factory(self):
        """Create a mock factory for testing."""
        factory = Mock(spec=LLMFactory)
        factory.detect_provider.return_value = "openai"
        return factory

    @pytest.fixture
    def mock_service(self):
        """Create a mock service for testing."""
        service = Mock()
        service.generate = AsyncMock()
        service.batch_generate = AsyncMock()
        service.get_available_models.return_value = ["gpt-4o", "gpt-4o-mini"]
        service.validate_connection = AsyncMock(return_value=True)
        return service

    @pytest.fixture
    def router(self, mock_factory):
        """Create a router with mock factory."""
        return LLMRouter(factory=mock_factory)

    @pytest.mark.asyncio
    async def test_generate_success(self, router, mock_factory, mock_service):
        """Test successful generation routing."""
        # Setup mocks
        mock_factory.get_service.return_value = mock_service
        expected_response = LLMResponse(
            content="Test response",
            model_used="gpt-4o",
            tokens_used=10,
            cost_estimate=0.001,
            latency_ms=100,
            provider="openai"
        )
        mock_service.generate.return_value = expected_response
        
        # Create request
        request = LLMRequest(
            model="gpt-4o",
            prompt="Test prompt",
            stream=False
        )
        
        # Test generation
        result = await router.generate(request)
        
        # Verify
        mock_factory.detect_provider.assert_called_once_with("gpt-4o")
        mock_factory.get_service.assert_called_once_with("openai")
        mock_service.generate.assert_called_once_with(request)
        assert result == expected_response

    @pytest.mark.asyncio
    async def test_generate_with_overrides(self, router, mock_factory, mock_service):
        """Test generation with parameter overrides."""
        mock_factory.get_service.return_value = mock_service
        mock_service.generate.return_value = Mock()
        
        request = LLMRequest(model="gpt-4o", prompt="Test", stream=False)
        overrides = {"temperature": 0.9, "max_tokens": 200}
        
        await router.generate(request, **overrides)
        
        mock_service.generate.assert_called_once_with(request, **overrides)

    @pytest.mark.asyncio
    async def test_generate_fallback_disabled(self, mock_factory, mock_service):
        """Test generation failure with fallback disabled."""
        # Create router with fallback disabled
        router = LLMRouter(factory=mock_factory, config={"enable_fallback": False})
        
        # Setup failure
        mock_factory.get_service.return_value = mock_service
        mock_service.generate.side_effect = Exception("API Error")
        
        request = LLMRequest(model="gpt-4o", prompt="Test", stream=False)
        
        # Should raise error without trying fallback
        with pytest.raises(Exception, match="Provider openai failed"):
            await router.generate(request)

    @pytest.mark.asyncio
    async def test_generate_fallback_success(self, mock_factory):
        """Test successful fallback to another provider."""
        # Create router with fallback enabled
        router = LLMRouter(
            factory=mock_factory,
            config={
                "enable_fallback": True,
                "fallback_providers": ["openai", "anthropic", "google"]
            }
        )
        
        # Primary service fails
        primary_service = Mock()
        primary_service.generate = AsyncMock(side_effect=Exception("Primary failed"))
        
        # Fallback service succeeds
        fallback_service = Mock()
        fallback_response = LLMResponse(
            content="Fallback response",
            model_used="claude-3-5-haiku-20241022",
            tokens_used=15,
            cost_estimate=0.002,
            latency_ms=200,
            provider="anthropic"
        )
        fallback_service.generate = AsyncMock(return_value=fallback_response)
        fallback_service.get_available_models.return_value = ["claude-3-5-haiku-20241022"]
        
        # Mock factory behavior
        def mock_get_service(provider):
            if provider == "openai":
                return primary_service
            elif provider == "anthropic":
                return fallback_service
            
        mock_factory.get_service.side_effect = mock_get_service
        
        request = LLMRequest(model="gpt-4o", prompt="Test", stream=False)
        
        result = await router.generate(request)
        
        assert result == fallback_response

    @pytest.mark.asyncio
    async def test_batch_generate_single_provider(self, router, mock_factory, mock_service):
        """Test batch generation with single provider."""
        # Setup
        mock_factory.detect_provider.return_value = "openai"
        mock_factory.get_service.return_value = mock_service
        
        responses = [
            LLMResponse(content="Response 1", model_used="gpt-4o", tokens_used=10, cost_estimate=0.001, latency_ms=100, provider="openai"),
            LLMResponse(content="Response 2", model_used="gpt-4o", tokens_used=12, cost_estimate=0.001, latency_ms=110, provider="openai")
        ]
        mock_service.batch_generate.return_value = responses
        
        requests = [
            LLMRequest(model="gpt-4o", prompt="Test 1", stream=False),
            LLMRequest(model="gpt-4o", prompt="Test 2", stream=False)
        ]
        
        result = await router.batch_generate(requests)
        
        assert len(result) == 2
        assert result[0] == responses[0]
        assert result[1] == responses[1]

    @pytest.mark.asyncio
    async def test_batch_generate_multiple_providers(self, mock_factory):
        """Test batch generation with multiple providers."""
        router = LLMRouter(factory=mock_factory)
        
        # Mock provider detection
        def mock_detect_provider(model):
            if model.startswith("gpt"):
                return "openai"
            elif model.startswith("claude"):
                return "anthropic"
        
        mock_factory.detect_provider.side_effect = mock_detect_provider
        
        # Mock services
        openai_service = Mock()
        openai_response = [LLMResponse(content="OpenAI", model_used="gpt-4o", tokens_used=10, cost_estimate=0.001, latency_ms=100, provider="openai")]
        openai_service.batch_generate = AsyncMock(return_value=openai_response)
        
        anthropic_service = Mock()
        anthropic_response = [LLMResponse(content="Anthropic", model_used="claude-3-5-haiku-20241022", tokens_used=12, cost_estimate=0.002, latency_ms=120, provider="anthropic")]
        anthropic_service.batch_generate = AsyncMock(return_value=anthropic_response)
        
        def mock_get_service(provider):
            if provider == "openai":
                return openai_service
            elif provider == "anthropic":
                return anthropic_service
        
        mock_factory.get_service.side_effect = mock_get_service
        
        requests = [
            LLMRequest(model="gpt-4o", prompt="Test 1", stream=False),
            LLMRequest(model="claude-3-5-haiku-20241022", prompt="Test 2", stream=False)
        ]
        
        result = await router.batch_generate(requests)
        
        assert len(result) == 2
        assert result[0].content == "OpenAI"
        assert result[1].content == "Anthropic"

    def test_get_model_info(self, router, mock_factory, mock_service):
        """Test getting model information."""
        mock_factory.get_service.return_value = mock_service
        mock_service.get_available_models.return_value = ["gpt-4o", "gpt-4o-mini"]
        
        # Mock string representation properly
        with patch.object(mock_service, '__str__', return_value="OpenAILLMService(model=gpt-4o-mini)"):
            info = router.get_model_info("gpt-4o")
        
        assert info["model"] == "gpt-4o"
        assert info["provider"] == "openai"
        assert info["available"] is True
        assert "OpenAILLMService" in info["service_info"]

    def test_get_model_info_unavailable(self, router, mock_factory, mock_service):
        """Test getting info for unavailable model."""
        mock_factory.get_service.return_value = mock_service
        mock_service.get_available_models.return_value = ["gpt-4o-mini"]  # gpt-4o not available
        
        info = router.get_model_info("gpt-4o")
        
        assert info["model"] == "gpt-4o"
        assert info["available"] is False

    def test_get_model_info_service_error(self, router, mock_factory):
        """Test getting model info when service creation fails."""
        mock_factory.get_service.side_effect = Exception("Service error")
        
        info = router.get_model_info("gpt-4o")
        
        assert info["model"] == "gpt-4o"
        assert info["available"] is False
        assert "error" in info
        assert "Service error" in info["error"]


class TestFactoryFunctions:
    """Test convenience factory functions."""

    def test_create_llm_router(self):
        """Test LLM router creation function."""
        router = create_llm_router()
        assert isinstance(router, LLMRouter)
        assert isinstance(router.factory, LLMFactory)

    def test_create_llm_router_with_config(self):
        """Test LLM router creation with config."""
        config = {"enable_fallback": False}
        router = create_llm_router(config=config)
        assert router.enable_fallback is False

    def test_create_llm_factory(self):
        """Test LLM factory creation function."""
        factory = create_llm_factory()
        assert isinstance(factory, LLMFactory)

    def test_create_llm_factory_with_config(self):
        """Test LLM factory creation with config."""
        config = {"providers": {}}
        factory = create_llm_factory(config=config)
        assert factory.config == config


class TestLLMRouterIntegration:
    """Integration tests for LLM Router with real service mocking."""

    @pytest.mark.asyncio
    async def test_router_with_streaming_request(self):
        """Test router handling streaming requests."""
        # Create router with real factory but mocked services
        router = create_llm_router()
        
        # Mock streaming response
        async def mock_streaming_generator():
            from src.models.streaming_models import StreamHandler, StreamingChunkType
            handler = StreamHandler("gpt-4o", "openai", "test")
            
            # Yield content chunk
            yield handler.process_chunk(
                StreamingChunkType.CONTENT,
                delta="Streaming response",
                metadata={}
            )
            
            # Yield done chunk
            final_response = handler.finalize(total_tokens=10, cost_estimate=0.001)
            yield handler.process_chunk(
                StreamingChunkType.DONE,
                metadata={"final_response": final_response.model_dump()}
            )
        
        # Mock the factory's service creation
        with patch.object(router.factory, 'get_service') as mock_get_service:
            mock_service = Mock()
            mock_service.generate = AsyncMock(return_value=mock_streaming_generator())
            mock_get_service.return_value = mock_service
            
            request = LLMRequest(
                model="gpt-4o",
                prompt="Test streaming",
                stream=True
            )
            
            result = await router.generate(request)
            
            # Collect streaming chunks
            chunks = []
            async for chunk in result:
                chunks.append(chunk)
            
            assert len(chunks) == 2  # content + done
            assert chunks[0].chunk_type.value == "content"
            assert chunks[1].chunk_type.value == "done"

    @pytest.mark.asyncio
    async def test_router_with_config_from_file(self):
        """Test router with configuration that matches config.yaml structure."""
        config = {
            "providers": {
                "openai": {
                    "default_model": "gpt-4o-mini",
                    "api_key_env": "OPENAI_API_KEY",
                    "timeout_seconds": 30,
                    "max_retries": 3
                },
                "anthropic": {
                    "default_model": "claude-3-5-haiku-20241022",
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "timeout_seconds": 30,
                    "max_retries": 3
                }
            },
            "enable_fallback": True,
            "fallback_providers": ["openai", "anthropic", "google"]
        }
        
        router = create_llm_router(config=config)
        
        assert router.enable_fallback is True
        assert router.fallback_providers == ["openai", "anthropic", "google"]
        assert "openai" in router.factory._provider_configs
        assert "anthropic" in router.factory._provider_configs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])