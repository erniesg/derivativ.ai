"""
Main Generation Service for orchestrating question generation pipeline.
Handles batch generation, coordination between agents, and database operations.
"""

import asyncio
import json
import time
import os
from typing import List, Dict, Any, Optional
from uuid import uuid4

from smolagents import LiteLLMModel, OpenAIServerModel, InferenceClientModel

from ..models import (
    GenerationRequest, GenerationResponse, GenerationConfig,
    CandidateQuestion, LLMModel, CalculatorPolicy, CommandWord
)
from ..database import NeonDBClient
from ..agents import QuestionGeneratorAgent
from ..services.config_manager import DEFAULT_PROMPT_TEMPLATE_VERSION_GENERATION


class QuestionGenerationService:
    """Main service for coordinating question generation"""

    def __init__(self, database_url: str = None, debug: bool = None):
        # For local testing, database_url can be None
        if database_url:
            self.db_client = NeonDBClient(database_url)
        else:
            # Use a dummy database URL for local testing
            self.db_client = NeonDBClient("postgresql://dummy:dummy@dummy/dummy")

        self.generators = {}  # Cache of generator agents by model

        # Set debug mode from parameter, environment variable, or default to False
        if debug is not None:
            self.debug = debug
        else:
            self.debug = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes", "on")

        # Load configuration
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load generation configuration"""
        try:
            with open("config/generation_config.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            # Fallback to default config
            return {
                "models": {
                    "generator": {"default": "gpt-4o"}
                },
                "generation_parameters": {
                    "temperature": 0.7,
                    "max_tokens": 4000
                }
            }

    async def initialize(self):
        """Initialize the service and database connections"""
        await self.db_client.connect()
        await self.db_client.create_candidate_questions_table()

    async def shutdown(self):
        """Clean shutdown of the service"""
        await self.db_client.close()

    def _get_generator_agent(self, model: LLMModel) -> QuestionGeneratorAgent:
        """Get or create a generator agent for the specified model"""
        if model.value not in self.generators:
            # Create LLM model based on provider using OpenAIServerModel consistently
            if model in [LLMModel.GPT_4O, LLMModel.GPT_4O_MINI]:
                # OpenAI models
                llm_model = OpenAIServerModel(
                    model_id=model.value,
                    api_base="https://api.openai.com/v1",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    temperature=self.config["generation_parameters"]["temperature"],
                    max_tokens=self.config["generation_parameters"]["max_tokens"],
                    response_format={"type": "json_object"}
                )
            elif model in [LLMModel.GEMINI_PRO, LLMModel.GEMINI_FLASH]:
                # Google Gemini using OpenAI-compatible API
                model_name = "gemini-2.0-flash" if model == LLMModel.GEMINI_FLASH else "gemini-pro"
                llm_model = OpenAIServerModel(
                    model_id=model_name,
                    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                    api_key=os.getenv("GOOGLE_API_KEY"),
                    temperature=self.config["generation_parameters"]["temperature"],
                    max_tokens=self.config["generation_parameters"]["max_tokens"],
                    response_format={"type": "json_object"}
                )
            elif model == LLMModel.CLAUDE_3_5_SONNET:
                # Anthropic Claude using LiteLLM (no OpenAI-compatible endpoint available)
                llm_model = LiteLLMModel(
                    model_id="anthropic/claude-3-5-sonnet-latest",
                    api_key=os.getenv("ANTHROPIC_API_KEY"),
                    temperature=self.config["generation_parameters"]["temperature"],
                    max_tokens=self.config["generation_parameters"]["max_tokens"],
                    response_format={"type": "json_object"}
                )
            elif model == LLMModel.CLAUDE_4_SONNET or model == LLMModel.CLAUDE_4_OPUS:
                # Claude 4 models via Amazon Bedrock
                from smolagents import AmazonBedrockServerModel
                llm_model = AmazonBedrockServerModel(
                    model_id=model.value,
                    client_kwargs={'region_name': os.getenv("AWS_REGION", "us-east-1")}
                )
            elif model.value == "deepseek-ai/DeepSeek-R1-0528":
                llm_model = InferenceClientModel(
                    model_id="deepseek-ai/DeepSeek-R1-0528",
                    provider="auto",  # Let HF choose the best available provider
                    token=os.getenv("HF_TOKEN"),
                    max_tokens=self.config["generation_parameters"]["max_tokens"]
                )
            elif model.value == "Qwen/Qwen3-235B-A22B":
                llm_model = InferenceClientModel(
                    model_id="Qwen/Qwen3-235B-A22B",
                    provider="auto",  # Use auto to fallback to any available provider
                    token=os.getenv("HF_TOKEN"),
                    max_tokens=self.config["generation_parameters"]["max_tokens"]
                )
            else:
                # Default to OpenAI GPT-4o
                llm_model = OpenAIServerModel(
                    model_id="gpt-4o",
                    api_base="https://api.openai.com/v1",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    temperature=self.config["generation_parameters"]["temperature"],
                    max_tokens=self.config["generation_parameters"]["max_tokens"],
                    response_format={"type": "json_object"}
                )

            self.generators[model.value] = QuestionGeneratorAgent(llm_model, self.db_client, debug=self.debug)

        return self.generators[model.value]

    async def generate_questions(self, request: GenerationRequest) -> GenerationResponse:
        """Generate candidate questions based on request"""
        start_time = time.time()
        generated_questions = []
        failed_generations = []

        total_requested = len(request.target_grades) * request.count_per_grade

        try:
            # Generate questions for each target grade
            for target_grade in request.target_grades:
                for i in range(request.count_per_grade):
                    try:
                        # Create generation config
                        config = self._create_generation_config(request, target_grade, i)

                        # Get appropriate generator agent
                        generator = self._get_generator_agent(config.llm_model_generation)

                        # Generate question
                        candidate_question = await generator.generate_question(config)

                        if candidate_question:
                            # Save to database
                            success = await self.db_client.save_candidate_question(candidate_question)
                            if success:
                                generated_questions.append(candidate_question)
                            else:
                                failed_generations.append({
                                    "config": config.model_dump(),
                                    "error": "Failed to save to database"
                                })
                        else:
                            failed_generations.append({
                                "config": config.model_dump(),
                                "error": "Question generation failed"
                            })

                    except Exception as e:
                        failed_generations.append({
                            "target_grade": target_grade,
                            "iteration": i,
                            "error": str(e)
                        })

        except Exception as e:
            # Log overall failure
            print(f"Generation service error: {e}")

        generation_time = time.time() - start_time

        return GenerationResponse(
            generated_questions=generated_questions,
            failed_generations=failed_generations,
            total_requested=total_requested,
            total_generated=len(generated_questions),
            generation_time_seconds=generation_time
        )

    def _create_generation_config(
        self,
        request: GenerationRequest,
        target_grade: int,
        iteration: int
    ) -> GenerationConfig:
        """Create a generation config for a specific question"""

        # Use provided config or create default
        if request.generation_config:
            base_config = request.generation_config.model_copy()
            base_config.target_grade = target_grade
            base_config.generation_id = uuid4()
            # Ensure prompt version is set
            if not base_config.prompt_template_version_generation:
                base_config.prompt_template_version_generation = DEFAULT_PROMPT_TEMPLATE_VERSION_GENERATION
            return base_config

        # Default subject content references based on grade level
        default_content_refs = self._get_default_content_refs(target_grade)
        content_refs = request.subject_content_references or default_content_refs

        return GenerationConfig(
            generation_id=uuid4(),
            seed_question_id=request.seed_question_id,
            target_grade=target_grade,
            calculator_policy=request.calculator_policy,
            desired_marks=min(target_grade // 2 + 1, 5),  # Scale marks with grade
            subject_content_references=content_refs,
            llm_model_generation=LLMModel(self.config["models"]["generator"]["default"]),
            temperature=self.config["generation_parameters"]["temperature"],
            max_tokens=self.config["generation_parameters"]["max_tokens"],
            prompt_template_version_generation=DEFAULT_PROMPT_TEMPLATE_VERSION_GENERATION
        )

    def _get_default_content_refs(self, target_grade: int) -> List[str]:
        """Get default content references based on target grade"""
        if target_grade <= 3:
            # Foundation topics
            return ["C1.6", "C1.4", "C1.5"]  # Basic operations, fractions, ordering
        elif target_grade <= 6:
            # Intermediate topics
            return ["C1.6", "C1.11", "C1.13"]  # Operations, ratio, percentages
        else:
            # Advanced topics
            return ["C1.7", "C1.8", "C2.1"]  # Indices, standard form, algebra

    async def generate_batch_from_seed(
        self,
        seed_question_id: str,
        target_grades: List[int],
        count_per_grade: int = 2
    ) -> GenerationResponse:
        """Generate a batch of questions inspired by a seed question"""

        request = GenerationRequest(
            seed_question_id=seed_question_id,
            target_grades=target_grades,
            count_per_grade=count_per_grade,
            calculator_policy=CalculatorPolicy.NOT_ALLOWED  # Default for safety
        )

        return await self.generate_questions(request)

    async def generate_by_topic(
        self,
        subject_content_references: List[str],
        target_grades: List[int],
        count_per_grade: int = 1,
        command_word: Optional[CommandWord] = None
    ) -> GenerationResponse:
        """Generate questions focused on specific syllabus topics"""

        # Create base config
        config = GenerationConfig(
            target_grade=target_grades[0],  # Will be overridden
            calculator_policy=CalculatorPolicy.NOT_ALLOWED,
            desired_marks=2,
            subject_content_references=subject_content_references,
            command_word_override=command_word,
            prompt_template_version_generation=DEFAULT_PROMPT_TEMPLATE_VERSION_GENERATION
        )

        request = GenerationRequest(
            target_grades=target_grades,
            count_per_grade=count_per_grade,
            subject_content_references=subject_content_references,
            generation_config=config
        )

        return await self.generate_questions(request)

    async def get_generation_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated questions"""
        return await self.db_client.get_generation_stats()

    async def get_candidate_questions(
        self,
        target_grade: Optional[int] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get candidate questions with optional filtering"""
        return await self.db_client.get_candidate_questions(
            target_grade=target_grade,
            limit=limit
        )

    async def update_question_review(
        self,
        generation_id: str,
        status: str,
        reviewer_notes: Optional[str] = None
    ) -> bool:
        """Update the review status of a candidate question"""
        from uuid import UUID
        from ..models.question_models import GenerationStatus

        try:
            gen_id = UUID(generation_id)
            gen_status = GenerationStatus(status)
            return await self.db_client.update_question_status(
                gen_id, gen_status, reviewer_notes
            )
        except Exception as e:
            print(f"Error updating question review: {e}")
            return False
