"""
Centralized configuration management system with environment variable support.
Supports YAML configuration, environment overrides, and runtime parameter management.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.models.llm_models import AppConfig

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages application configuration with support for:
    - YAML file loading
    - Environment variable overrides
    - Runtime configuration updates
    - Default value fallbacks
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to YAML configuration file. If None, uses default locations.
        """
        self.config_path = self._resolve_config_path(config_path)
        self._config_cache: Optional[AppConfig] = None
        self._env_overrides = self._load_env_overrides()

    def _resolve_config_path(self, config_path: Optional[Union[str, Path]]) -> Optional[Path]:
        """Resolve configuration file path."""
        if config_path:
            return Path(config_path)

        # Try default locations
        default_paths = [
            Path("config.yaml"),
            Path("config/config.yaml"),
            Path("src/config/config.yaml"),
        ]

        for path in default_paths:
            if path.exists():
                return path

        logger.warning("No configuration file found, using defaults")
        return None

    def _load_env_overrides(self) -> dict[str, Any]:
        """Load configuration overrides from environment variables."""
        overrides = {}

        # LLM defaults overrides
        if "LLM_DEFAULT_TEMPERATURE" in os.environ:
            overrides.setdefault("llm_defaults", {})["temperature"] = float(
                os.environ["LLM_DEFAULT_TEMPERATURE"]
            )

        if "LLM_DEFAULT_MAX_TOKENS" in os.environ:
            overrides.setdefault("llm_defaults", {})["max_tokens"] = int(
                os.environ["LLM_DEFAULT_MAX_TOKENS"]
            )

        if "LLM_DEFAULT_TOP_P" in os.environ:
            overrides.setdefault("llm_defaults", {})["top_p"] = float(
                os.environ["LLM_DEFAULT_TOP_P"]
            )

        if "LLM_DEFAULT_TIMEOUT" in os.environ:
            overrides.setdefault("llm_defaults", {})["timeout_seconds"] = int(
                os.environ["LLM_DEFAULT_TIMEOUT"]
            )

        # Provider overrides
        if "LLM_PROVIDERS_DEFAULT_PROVIDER" in os.environ:
            overrides.setdefault("llm_providers", {})["default_provider"] = os.environ[
                "LLM_PROVIDERS_DEFAULT_PROVIDER"
            ]

        if "LLM_PROVIDERS_ENABLE_FALLBACK" in os.environ:
            overrides.setdefault("llm_providers", {})["enable_fallback"] = (
                os.environ["LLM_PROVIDERS_ENABLE_FALLBACK"].lower() == "true"
            )

        return overrides

    def _load_yaml_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path or not self.config_path.exists():
            logger.info("Using default configuration")
            return self._get_default_config()

        try:
            with open(self.config_path, encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            logger.info(f"Loaded configuration from {self.config_path}")
            return config_data or {}

        except Exception as e:
            logger.error(f"Failed to load configuration from {self.config_path}: {e}")
            logger.info("Falling back to default configuration")
            return self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration when no file is available."""
        return {
            "llm_providers": {
                "openai": {
                    "default_model": "gpt-4.1-nano",
                    "api_key_env": "OPENAI_API_KEY",
                    "base_url": "https://api.openai.com/v1",
                },
                "anthropic": {
                    "default_model": "claude-3-5-haiku-20241022",
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "base_url": "https://api.anthropic.com",
                },
                "google": {
                    "default_model": "gemini-2.5-flash-lite-preview-06-17",
                    "api_key_env": "GOOGLE_API_KEY",
                },
                "default_provider": "openai",
                "enable_fallback": True,
                "cost_tracking": True,
            },
            "llm_defaults": {
                "temperature": 0.7,
                "max_tokens": 1000,
                "top_p": 0.9,
                "timeout_seconds": 30,
            },
            "quality_control": {
                "thresholds": {
                    "auto_approve": 0.85,
                    "manual_review": 0.70,
                    "refine": 0.60,
                    "regenerate": 0.40,
                    "reject": 0.20,
                }
            },
            "agents": {
                "question_generator": {
                    "model": "gpt-4.1-nano",
                    "temperature": 0.8,
                    "max_tokens": 2000,
                    "extra_params": {"frequency_penalty": 0.1},
                },
                "marker": {
                    "model": "claude-3-5-haiku-20241022",
                    "temperature": 0.3,
                    "max_tokens": 1500,
                    "extra_params": {"top_k": 40},
                },
                "reviewer": {
                    "model": "claude-3-5-haiku-20241022",
                    "temperature": 0.5,
                    "max_tokens": 1000,
                    "extra_params": {"top_k": 20},
                },
                "refinement": {
                    "model": "gpt-4.1-nano",
                    "temperature": 0.7,
                    "max_tokens": 2500,
                    "extra_params": {"frequency_penalty": 0.2},
                },
            },
            "diagram_generation": {
                "storage": {
                    "type": "local",
                    "base_path": "generated_diagrams/",
                    "auto_cleanup": False,
                },
                "rendering": {
                    "quality": "low",  # low, medium, high
                    "timeout_seconds": 60,
                    "manim_flags": ["-ql"],  # Low quality for speed
                },
                "quality_control": {
                    "min_quality_threshold": 0.8,
                    "max_retry_attempts": 3,
                    "auto_approve_threshold": 0.9,
                },
                "auto_detection": {
                    "enabled": True,
                    "geometry_keywords": [
                        "triangle",
                        "quadrilateral",
                        "circle",
                        "angle",
                        "parallel",
                        "perpendicular",
                        "polygon",
                        "vertex",
                        "vertices",
                        "diagram",
                    ],
                    "exclude_keywords": ["not shown", "no diagram", "text only"],
                },
            },
        }

    def _merge_configs(self, base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
        """Deep merge configuration dictionaries."""
        result = base.copy()

        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def load_config(self, force_reload: bool = False) -> AppConfig:
        """
        Load and return application configuration.

        Args:
            force_reload: If True, force reload from file even if cached

        Returns:
            Validated AppConfig instance
        """
        if self._config_cache is not None and not force_reload:
            return self._config_cache

        # Load base configuration
        yaml_config = self._load_yaml_config()

        # Apply environment overrides
        final_config = self._merge_configs(yaml_config, self._env_overrides)

        try:
            # Validate and create config object
            self._config_cache = AppConfig(**final_config)
            logger.info("Configuration loaded and validated successfully")
            return self._config_cache

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            # Fall back to default config
            default_config = self._get_default_config()
            self._config_cache = AppConfig(**default_config)
            logger.info("Using validated default configuration")
            return self._config_cache

    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a provider from environment variables.

        Args:
            provider: Provider name (openai, anthropic, google)

        Returns:
            API key if found, None otherwise
        """
        config = self.load_config()
        provider_config = getattr(config.llm_providers, provider, None)

        if not provider_config:
            logger.error(f"Unknown provider: {provider}")
            return None

        api_key = os.environ.get(provider_config.api_key_env)
        if not api_key:
            logger.warning(
                f"API key not found for {provider} (env var: {provider_config.api_key_env})"
            )

        return api_key

    def update_config(self, updates: dict[str, Any], persist: bool = False) -> AppConfig:
        """
        Update configuration at runtime.

        Args:
            updates: Configuration updates to apply
            persist: If True, save changes to config file

        Returns:
            Updated AppConfig instance
        """
        current_config = self.load_config().model_dump()
        updated_config = self._merge_configs(current_config, updates)

        # Validate updated config
        self._config_cache = AppConfig(**updated_config)

        if persist and self.config_path:
            try:
                with open(self.config_path, "w", encoding="utf-8") as f:
                    yaml.dump(updated_config, f, default_flow_style=False, indent=2)
                logger.info(f"Configuration saved to {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to save configuration: {e}")

        return self._config_cache

    def reload_config(self) -> AppConfig:
        """Reload configuration from file and environment."""
        self._env_overrides = self._load_env_overrides()
        return self.load_config(force_reload=True)


class _ConfigManagerSingleton:
    """Singleton holder for ConfigManager."""

    _instance: Optional[ConfigManager] = None

    @classmethod
    def get_instance(cls, config_path: Optional[Union[str, Path]] = None) -> ConfigManager:
        """Get or create ConfigManager instance."""
        if cls._instance is None:
            cls._instance = ConfigManager(config_path)
        return cls._instance


def get_config_manager(config_path: Optional[Union[str, Path]] = None) -> ConfigManager:
    """
    Get global configuration manager instance.

    Args:
        config_path: Path to config file (only used on first call)

    Returns:
        ConfigManager instance
    """
    return _ConfigManagerSingleton.get_instance(config_path)


def load_config(config_path: Optional[Union[str, Path]] = None) -> AppConfig:
    """
    Load application configuration.

    Args:
        config_path: Path to config file

    Returns:
        AppConfig instance
    """
    manager = get_config_manager(config_path)
    return manager.load_config()


def get_api_key(provider: str) -> Optional[str]:
    """
    Get API key for a provider.

    Args:
        provider: Provider name

    Returns:
        API key if found
    """
    manager = get_config_manager()
    return manager.get_api_key(provider)


class Settings(BaseSettings):
    """
    Additional settings managed through environment variables.
    Used for sensitive data and deployment-specific configuration.
    """

    # Database
    database_url: Optional[str] = Field(
        None, description="Database connection URL", alias="DATABASE_URL"
    )

    # API Keys (redundant with ConfigManager, but useful for validation)
    openai_api_key: Optional[str] = Field(
        None, description="OpenAI API key", alias="OPENAI_API_KEY"
    )
    anthropic_api_key: Optional[str] = Field(
        None, description="Anthropic API key", alias="ANTHROPIC_API_KEY"
    )
    google_api_key: Optional[str] = Field(
        None, description="Google API key", alias="GOOGLE_API_KEY"
    )

    # Development settings
    debug: bool = Field(False, description="Enable debug mode", alias="DEBUG")
    log_level: str = Field("INFO", description="Logging level", alias="LOG_LEVEL")

    # Performance settings
    max_concurrent_requests: int = Field(
        10, description="Maximum concurrent requests", alias="MAX_CONCURRENT_REQUESTS"
    )
    request_timeout: int = Field(
        30, description="Request timeout in seconds", alias="REQUEST_TIMEOUT"
    )

    # Demo/testing settings
    demo_mode: bool = Field(False, description="Enable demo mode", alias="DEMO_MODE")
    use_mock_llm: bool = Field(False, description="Use mock LLM service", alias="USE_MOCK_LLM")

    # Diagram storage settings
    diagram_storage_type: str = Field(
        "local",
        description="Diagram storage type: local, supabase, s3",
        alias="DIAGRAM_STORAGE_TYPE",
    )
    diagram_base_path: str = Field(
        "generated_diagrams/",
        description="Base path for diagram storage",
        alias="DIAGRAM_BASE_PATH",
    )
    diagram_auto_detect: bool = Field(
        True, description="Auto-detect when diagrams are needed", alias="DIAGRAM_AUTO_DETECT"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="",
        # Map field names to environment variable names
        env_ignore_empty=True,
    )


class _SettingsSingleton:
    """Singleton holder for Settings."""

    _instance: Optional[Settings] = None

    @classmethod
    def get_instance(cls) -> Settings:
        """Get or create Settings instance."""
        if cls._instance is None:
            cls._instance = Settings()
        return cls._instance


def get_settings() -> Settings:
    """Get global settings instance."""
    return _SettingsSingleton.get_instance()


def validate_api_keys() -> dict[str, bool]:
    """
    Validate that required API keys are available.

    Returns:
        Dict mapping provider names to availability status
    """
    config = load_config()
    status = {}

    for provider_name in ["openai", "anthropic", "google"]:
        provider_config = getattr(config.llm_providers, provider_name)
        api_key = os.environ.get(provider_config.api_key_env)
        status[provider_name] = api_key is not None and len(api_key.strip()) > 0

    return status


def get_model_for_agent(agent_name: str, provider_override: Optional[str] = None) -> str:
    """
    Get the appropriate model for an agent.

    Args:
        agent_name: Name of the agent
        provider_override: Optional provider override

    Returns:
        Model name to use
    """
    config = load_config()

    # Check agent-specific model configuration
    agent_config = config.agents.get(agent_name, {})
    if "model" in agent_config:
        return agent_config["model"]

    # Use provider override or default provider
    provider = provider_override or config.llm_providers.default_provider
    provider_config = getattr(config.llm_providers, provider)

    return provider_config.default_model
