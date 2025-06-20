"""
Pydantic models for LLM configuration and request/response handling.
Provides type safety, validation, and seamless integration with Jinja templates.
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class LLMRequest(BaseModel):
    """
    Universal LLM request model that works across all providers.
    Supports template variables and runtime overrides.
    """

    model: str = Field(..., description="Model name to use")
    prompt: str = Field(..., description="The prompt text")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(1000, gt=0, le=100000, description="Maximum tokens to generate")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    system_message: Optional[str] = Field(None, description="System/instruction message")
    stop_sequences: list[str] = Field(default_factory=list, description="Stop sequences")
    stream: bool = Field(True, description="Enable streaming responses by default")
    extra_params: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific parameters"
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v):
        """Ensure prompt is not empty after stripping whitespace."""
        if not v or not v.strip():
            raise ValueError("prompt cannot be empty")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        """Ensure model name is not empty."""
        if not v or not v.strip():
            raise ValueError("model cannot be empty")
        return v

    model_config = ConfigDict(
        # Allow extra fields for forward compatibility
        extra="ignore",
        # Enable JSON schema generation
        json_schema_extra={
            "example": {
                "model": "gpt-4.1-nano",
                "prompt": "Generate a {{question_type}} question about {{topic}}",
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": True,
                "system_message": "You are a {{grade}} grade {{subject}} teacher",
                "extra_params": {"frequency_penalty": 0.1, "thinking": {"budget_tokens": 1024}},
            }
        },
    )


class LLMResponse(BaseModel):
    """
    Universal LLM response model with normalized fields across providers.
    """

    content: str = Field(..., description="Generated content")
    model_used: str = Field(..., description="Actual model that generated the response")
    tokens_used: int = Field(..., ge=0, description="Number of tokens consumed")
    cost_estimate: float = Field(..., ge=0.0, description="Estimated cost in USD")
    latency_ms: int = Field(..., ge=0, description="Response latency in milliseconds")
    provider: str = Field(..., description="Provider that generated the response")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional response metadata"
    )

    @field_validator("tokens_used")
    @classmethod
    def validate_tokens_used(cls, v):
        """Ensure tokens_used is non-negative."""
        if v < 0:
            raise ValueError("tokens_used must be non-negative")
        return v

    @field_validator("cost_estimate")
    @classmethod
    def validate_cost_estimate(cls, v):
        """Ensure cost_estimate is non-negative."""
        if v < 0:
            raise ValueError("cost_estimate must be non-negative")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "What is the value of x in the equation 2x + 5 = 13?",
                "model_used": "gpt-4.1-nano",
                "tokens_used": 25,
                "cost_estimate": 0.000012,
                "latency_ms": 800,
                "provider": "openai",
                "metadata": {"finish_reason": "stop", "prompt_tokens": 20, "completion_tokens": 25},
            }
        }
    )


class LLMProviderConfig(BaseModel):
    """Configuration for a specific LLM provider."""

    default_model: str = Field(..., description="Default model for this provider")
    api_key_env: str = Field(..., description="Environment variable name for API key")
    base_url: Optional[str] = Field(None, description="Base URL for API endpoints")
    api_version: Optional[str] = Field(None, description="API version to use")
    timeout_seconds: int = Field(30, gt=0, description="Request timeout in seconds")
    max_retries: int = Field(3, ge=0, description="Maximum number of retries")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "default_model": "gpt-4.1-nano",
                "api_key_env": "OPENAI_API_KEY",
                "base_url": "https://api.openai.com/v1",
                "timeout_seconds": 30,
                "max_retries": 3,
            }
        }
    )


class LLMProvidersConfig(BaseModel):
    """Configuration for all LLM providers."""

    openai: LLMProviderConfig = Field(..., description="OpenAI configuration")
    anthropic: LLMProviderConfig = Field(..., description="Anthropic configuration")
    google: LLMProviderConfig = Field(..., description="Google configuration")
    default_provider: LLMProvider = Field(..., description="Default provider to use")
    enable_fallback: bool = Field(True, description="Enable provider fallback on failure")
    cost_tracking: bool = Field(True, description="Track usage costs")

    @field_validator("default_provider")
    @classmethod
    def validate_default_provider(cls, v):
        """Ensure default provider is one of the supported providers."""
        if v not in LLMProvider.__members__.values():
            valid_providers = ", ".join(LLMProvider.__members__.values())
            raise ValueError(f"default_provider must be one of: {valid_providers}")
        return v

    model_config = ConfigDict(
        use_enum_values=True  # Use enum values in serialization
    )


class LLMDefaults(BaseModel):
    """Default LLM parameters used across all requests."""

    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Default temperature")
    max_tokens: int = Field(1000, gt=0, le=100000, description="Default max tokens")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Default top_p")
    timeout_seconds: int = Field(30, gt=0, description="Default request timeout")

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        """Ensure temperature is in valid range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, v):
        """Ensure top_p is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v):
        """Ensure max_tokens is positive and reasonable."""
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        if v > 100000:
            raise ValueError("max_tokens must not exceed 100000")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"temperature": 0.7, "max_tokens": 1000, "top_p": 0.9, "timeout_seconds": 30}
        }
    )


class AgentConfig(BaseModel):
    """Configuration for individual agents with LLM settings."""

    model: Optional[str] = Field(None, description="Agent-specific model override")
    temperature: Optional[float] = Field(
        None, ge=0.0, le=2.0, description="Agent-specific temperature"
    )
    max_tokens: Optional[int] = Field(
        None, gt=0, le=100000, description="Agent-specific max tokens"
    )
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Agent-specific top_p")
    system_message: Optional[str] = Field(None, description="Agent-specific system message")
    extra_params: dict[str, Any] = Field(
        default_factory=dict, description="Agent-specific extra parameters"
    )

    def merge_with_defaults(self, defaults: LLMDefaults) -> dict[str, Any]:
        """Merge agent config with global defaults, agent config takes precedence."""
        merged = {
            "temperature": self.temperature
            if self.temperature is not None
            else defaults.temperature,
            "max_tokens": self.max_tokens if self.max_tokens is not None else defaults.max_tokens,
            "top_p": self.top_p if self.top_p is not None else defaults.top_p,
            "extra_params": self.extra_params,
        }

        if self.model is not None:
            merged["model"] = self.model
        if self.system_message is not None:
            merged["system_message"] = self.system_message

        return merged


class AppConfig(BaseModel):
    """Complete application configuration including LLM and other settings."""

    llm_providers: LLMProvidersConfig = Field(..., description="LLM provider configurations")
    llm_defaults: LLMDefaults = Field(..., description="Default LLM parameters")

    # Existing configuration sections (as Dict for flexibility)
    quality_control: dict[str, Any] = Field(
        default_factory=dict, description="Quality control settings"
    )
    agents: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Agent configurations"
    )
    performance: dict[str, Any] = Field(default_factory=dict, description="Performance settings")
    cambridge: dict[str, Any] = Field(default_factory=dict, description="Cambridge IGCSE settings")
    demo: dict[str, Any] = Field(default_factory=dict, description="Demo settings")
    logging: dict[str, Any] = Field(default_factory=dict, description="Logging settings")
    modal: dict[str, Any] = Field(default_factory=dict, description="Modal deployment settings")
    templates: dict[str, Any] = Field(default_factory=dict, description="Template settings")

    def get_agent_llm_config(self, agent_name: str) -> dict[str, Any]:
        """
        Get complete LLM configuration for a specific agent.
        Merges global defaults with agent-specific overrides.
        """
        agent_config = self.agents.get(agent_name, {})

        # Start with global defaults
        config = {
            "temperature": self.llm_defaults.temperature,
            "max_tokens": self.llm_defaults.max_tokens,
            "top_p": self.llm_defaults.top_p,
            "extra_params": {},
        }

        # Apply agent-specific overrides
        for key in ["model", "temperature", "max_tokens", "top_p", "system_message"]:
            if key in agent_config:
                config[key] = agent_config[key]

        # Merge extra_params
        if "extra_params" in agent_config:
            config["extra_params"] = agent_config["extra_params"]

        return config

    def create_llm_request_for_agent(self, agent_name: str, prompt: str, **overrides) -> LLMRequest:
        """
        Create an LLM request for a specific agent using its configuration.

        Args:
            agent_name: Name of the agent
            prompt: The prompt text
            **overrides: Runtime overrides for any parameter

        Returns:
            LLMRequest configured for the agent with any overrides applied
        """
        # Get agent's LLM configuration
        agent_config = self.get_agent_llm_config(agent_name)

        # Apply runtime overrides
        final_config = {**agent_config, **overrides}

        # Use default model if none specified
        if "model" not in final_config:
            provider = self.llm_providers.default_provider
            provider_config = getattr(self.llm_providers, provider)
            final_config["model"] = provider_config.default_model

        return LLMRequest(prompt=prompt, **final_config)

    model_config = ConfigDict(
        # Enable JSON schema generation
        json_schema_extra={
            "example": {
                "llm_providers": {
                    "openai": {"default_model": "gpt-4.1-nano", "api_key_env": "OPENAI_API_KEY"},
                    "anthropic": {
                        "default_model": "claude-3-5-haiku-20241022",
                        "api_key_env": "ANTHROPIC_API_KEY",
                    },
                    "google": {
                        "default_model": "gemini-2.5-flash-lite-preview-06-17",
                        "api_key_env": "GOOGLE_API_KEY",
                    },
                    "default_provider": "openai",
                },
                "llm_defaults": {
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "top_p": 0.9,
                    "timeout_seconds": 30,
                },
                "agents": {
                    "question_generator": {
                        "model": "gpt-4.1-nano",
                        "temperature": 0.8,
                        "extra_params": {"frequency_penalty": 0.1},
                    }
                },
            }
        }
    )


# Utility functions for working with LLM models


def create_llm_request_from_template_vars(
    template_vars: dict[str, Any], model: str, **overrides
) -> LLMRequest:
    """
    Create an LLM request from template variables.

    Args:
        template_vars: Variables from template processing
        model: Model name to use
        **overrides: Additional parameters to override

    Returns:
        Configured LLMRequest
    """
    # Extract standard parameters from template vars
    request_params = {
        "model": model,
        "prompt": template_vars.get("prompt", ""),
        "temperature": template_vars.get("temperature", 0.7),
        "max_tokens": template_vars.get("max_tokens", 1000),
        "top_p": template_vars.get("top_p", 0.9),
        "system_message": template_vars.get("system_message"),
        "stop_sequences": template_vars.get("stop_sequences", []),
        "extra_params": template_vars.get("extra_params", {}),
    }

    # Apply overrides
    request_params.update(overrides)

    return LLMRequest(**request_params)


def merge_llm_configs(base_config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """
    Merge LLM configurations with proper handling of nested extra_params.

    Args:
        base_config: Base configuration
        overrides: Override configuration

    Returns:
        Merged configuration
    """
    merged = base_config.copy()

    for key, value in overrides.items():
        if key == "extra_params" and key in merged:
            # Deep merge extra_params
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value

    return merged
