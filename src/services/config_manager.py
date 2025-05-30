"""
Configuration Manager for Question Generation System.
Handles loading and managing different generation configurations for various question types and models.
"""

import json
import os
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..models import GenerationConfig, LLMModel, CalculatorPolicy, CommandWord


@dataclass
class QuestionGenerationConfigTemplate:
    """Template for question generation configuration"""
    config_id: str
    description: str
    target_grades: List[int]
    subject_content_references: List[str]
    calculator_policy: str
    desired_marks: int
    llm_model_generation: str
    llm_model_marking_scheme: str
    llm_model_review: str
    prompt_template_version_generation: str
    prompt_template_version_marking_scheme: str
    prompt_template_version_review: str
    temperature: float
    max_tokens: int
    command_words: List[str]


@dataclass
class BatchConfig:
    """Configuration for batch generation"""
    description: str
    configs_to_use: List[str]
    questions_per_config: int
    total_questions: int


class ConfigManager:
    """Manages generation configurations and creates GenerationConfig objects"""

    def __init__(self, config_file_path: str = "config/question_generation_configs.json"):
        self.config_file_path = config_file_path
        self.configs: Dict[str, QuestionGenerationConfigTemplate] = {}
        self.batch_configs: Dict[str, BatchConfig] = {}
        self._load_configs()

    def _load_configs(self):
        """Load configurations from JSON file"""
        try:
            with open(self.config_file_path, 'r') as f:
                data = json.load(f)

            # Load individual configs
            for config_id, config_data in data.get("configs", {}).items():
                self.configs[config_id] = QuestionGenerationConfigTemplate(**config_data)

            # Load batch configs
            for batch_id, batch_data in data.get("batch_configs", {}).items():
                self.batch_configs[batch_id] = BatchConfig(**batch_data)

            print(f"Loaded {len(self.configs)} individual configs and {len(self.batch_configs)} batch configs")

        except FileNotFoundError:
            print(f"Warning: Config file {self.config_file_path} not found. Using empty configs.")
        except Exception as e:
            print(f"Error loading configs: {e}")

    def get_config_template(self, config_id: str) -> Optional[QuestionGenerationConfigTemplate]:
        """Get a configuration template by ID"""
        return self.configs.get(config_id)

    def list_available_configs(self) -> List[str]:
        """List all available configuration IDs"""
        return list(self.configs.keys())

    def list_batch_configs(self) -> List[str]:
        """List all available batch configuration IDs"""
        return list(self.batch_configs.keys())

    def create_generation_config(
        self,
        config_id: str,
        target_grade: Optional[int] = None,
        seed_question_id: Optional[str] = None,
        override_params: Optional[Dict[str, Any]] = None
    ) -> Optional[GenerationConfig]:
        """Create a GenerationConfig from a template"""
        template = self.get_config_template(config_id)
        if not template:
            print(f"Config template '{config_id}' not found")
            return None

        try:
            # Select target grade
            if target_grade is None:
                target_grade = random.choice(template.target_grades)
            elif target_grade not in template.target_grades:
                print(f"Warning: Target grade {target_grade} not in recommended grades {template.target_grades}")

            # Select command word
            command_word = None
            if template.command_words:
                try:
                    command_word_str = random.choice(template.command_words)
                    command_word = CommandWord(command_word_str)
                except ValueError:
                    print(f"Warning: Invalid command word '{command_word_str}', using None")

            # Create config - handle string models for HF models
            llm_model_generation = template.llm_model_generation
            llm_model_marking_scheme = template.llm_model_marking_scheme
            llm_model_review = template.llm_model_review

            # Try to convert to enum, but keep string if not found (for HF models)
            try:
                llm_model_generation = LLMModel(template.llm_model_generation)
            except ValueError:
                pass

            try:
                llm_model_marking_scheme = LLMModel(template.llm_model_marking_scheme)
            except ValueError:
                pass

            try:
                llm_model_review = LLMModel(template.llm_model_review)
            except ValueError:
                pass

            config = GenerationConfig(
                seed_question_id=seed_question_id,
                target_grade=target_grade,
                calculator_policy=CalculatorPolicy(template.calculator_policy),
                desired_marks=template.desired_marks,
                subject_content_references=template.subject_content_references,
                command_word_override=command_word,
                llm_model_generation=llm_model_generation,
                llm_model_marking_scheme=llm_model_marking_scheme,
                llm_model_review=llm_model_review,
                prompt_template_version_generation=template.prompt_template_version_generation,
                prompt_template_version_marking_scheme=template.prompt_template_version_marking_scheme,
                prompt_template_version_review=template.prompt_template_version_review,
                temperature=template.temperature,
                max_tokens=template.max_tokens
            )

            # Apply any overrides
            if override_params:
                for key, value in override_params.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

            return config

        except Exception as e:
            print(f"Error creating generation config: {e}")
            return None

    def create_batch_generation_configs(
        self,
        batch_config_id: str,
        seed_question_id: Optional[str] = None
    ) -> List[GenerationConfig]:
        """Create multiple GenerationConfigs from a batch configuration"""
        batch_config = self.batch_configs.get(batch_config_id)
        if not batch_config:
            print(f"Batch config '{batch_config_id}' not found")
            return []

        configs = []
        for config_id in batch_config.configs_to_use:
            for _ in range(batch_config.questions_per_config):
                config = self.create_generation_config(
                    config_id=config_id,
                    seed_question_id=seed_question_id
                )
                if config:
                    configs.append(config)

        return configs

    def get_configs_for_grade_range(self, min_grade: int, max_grade: int) -> List[str]:
        """Get config IDs that target questions within a grade range"""
        matching_configs = []
        for config_id, template in self.configs.items():
            if any(min_grade <= grade <= max_grade for grade in template.target_grades):
                matching_configs.append(config_id)
        return matching_configs

    def get_configs_by_model(self, model_name: str) -> List[str]:
        """Get config IDs that use a specific model"""
        matching_configs = []
        for config_id, template in self.configs.items():
            if (template.llm_model_generation == model_name or
                template.llm_model_marking_scheme == model_name or
                template.llm_model_review == model_name):
                matching_configs.append(config_id)
        return matching_configs

    def get_config_info(self, config_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a configuration"""
        template = self.get_config_template(config_id)
        if not template:
            return None

        return {
            "config_id": template.config_id,
            "description": template.description,
            "target_grades": template.target_grades,
            "subject_areas": template.subject_content_references,
            "calculator_policy": template.calculator_policy,
            "marks": template.desired_marks,
            "models": {
                "generation": template.llm_model_generation,
                "marking": template.llm_model_marking_scheme,
                "review": template.llm_model_review
            },
            "prompts": {
                "generation": template.prompt_template_version_generation,
                "marking": template.prompt_template_version_marking_scheme,
                "review": template.prompt_template_version_review
            },
            "parameters": {
                "temperature": template.temperature,
                "max_tokens": template.max_tokens
            },
            "command_words": template.command_words
        }
