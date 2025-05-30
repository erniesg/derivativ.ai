#!/usr/bin/env python3
"""
Candidate Question Generation Script

This script generates IGCSE Mathematics candidate questions using the configured
generation pipeline with smolagents. It supports single config generation,
batch generation, and saves all results to the Neon database.

Usage:
    python scripts/generate_candidate_questions.py --config basic_arithmetic_gpt4o
    python scripts/generate_candidate_questions.py --batch comprehensive_review
    python scripts/generate_candidate_questions.py --config algebra_claude4 --grade 5 --count 3
"""

import asyncio
import argparse
import os
import sys
import time
from typing import List, Optional

# Add project root to Python path for clean imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Clean imports from reorganized src package
from src.services import ConfigManager
from src.database import NeonDBClient
from src.agents import QuestionGeneratorAgent
from src.models import CandidateQuestion, GenerationConfig

from smolagents import OpenAIServerModel, LiteLLMModel, AmazonBedrockServerModel

from dotenv import load_dotenv
load_dotenv()


class CandidateQuestionGenerator:
    """Main generator for candidate questions using configurations"""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.db_client = NeonDBClient()

    async def setup(self):
        """Initialize database connection"""
        await self.db_client.connect()
        await self.db_client.create_candidate_questions_table()

    async def cleanup(self):
        """Clean up connections"""
        await self.db_client.close()

    def _create_model(self, model_name: str):
        """Create appropriate model instance based on model name"""
        try:
            if model_name.startswith('gpt-'):
                return OpenAIServerModel(
                    model_id=model_name,
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            elif model_name.startswith('anthropic.claude') or model_name.startswith('us.anthropic.claude'):
                # Claude 4 models via Amazon Bedrock
                return AmazonBedrockServerModel(
                    model_id=model_name,
                    client_kwargs={'region_name': os.getenv("AWS_REGION", "us-east-1")}
                )
            elif model_name == 'claude-4-sonnet' or model_name == 'claude-4-opus':
                # Legacy names - map to Bedrock model IDs
                bedrock_model_id = (
                    "us.anthropic.claude-sonnet-4-20250514-v1:0" if "sonnet" in model_name
                    else "us.anthropic.claude-opus-4-20250514-v1:0"
                )
                return AmazonBedrockServerModel(
                    model_id=bedrock_model_id,
                    client_kwargs={'region_name': os.getenv("AWS_REGION", "us-east-1")}
                )
            elif model_name == 'gemini-pro':
                return OpenAIServerModel(
                    model_id="gemini-pro",
                    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                    api_key=os.getenv("GOOGLE_API_KEY")
                )
            elif model_name.startswith('deepseek-ai/') or model_name.startswith('Qwen/'):
                # HuggingFace models
                return OpenAIServerModel(
                    model_id=model_name,
                    api_base="https://api-inference.huggingface.co/models",
                    api_key=os.getenv("HF_TOKEN")
                )
            else:
                # Fallback to LiteLLM for other models
                return LiteLLMModel(model_id=model_name)
        except Exception as e:
            print(f"Error creating model {model_name}: {e}")
            # Fallback to GPT-4o-mini
            return OpenAIServerModel(
                model_id="gpt-4o-mini",
                api_key=os.getenv("OPENAI_API_KEY")
            )

    async def generate_single_question(
        self,
        config_id: str,
        target_grade: Optional[int] = None,
        seed_question_id: Optional[str] = None
    ) -> Optional[CandidateQuestion]:
        """Generate a single candidate question using a specific config"""

        print(f"\n🔄 Generating question with config: {config_id}")

        # Create generation config
        generation_config = self.config_manager.create_generation_config(
            config_id=config_id,
            target_grade=target_grade,
            seed_question_id=seed_question_id
        )

        if not generation_config:
            print(f"❌ Failed to create generation config for {config_id}")
            return None

        print(f"   Target Grade: {generation_config.target_grade}")
        print(f"   Subject Areas: {', '.join(generation_config.subject_content_references)}")
        print(f"   Model: {generation_config.llm_model_generation}")
        print(f"   Marks: {generation_config.desired_marks}")

        # Create model and agent
        model = self._create_model(generation_config.llm_model_generation)
        agent = QuestionGeneratorAgent(model=model, db_client=self.db_client, debug=True)

        # Generate question
        start_time = time.time()
        candidate_question = await agent.generate_question(generation_config)
        generation_time = time.time() - start_time

        if candidate_question:
            # Save to database
            success = await self.db_client.save_candidate_question(candidate_question)
            if success:
                print(f"✅ Question generated and saved in {generation_time:.2f}s")
                print(f"   Question ID: {candidate_question.question_id_global}")
                print(f"   Content preview: {candidate_question.raw_text_content[:100]}...")
                return candidate_question
            else:
                print(f"❌ Failed to save question to database")
        else:
            print(f"❌ Failed to generate question with {config_id}")

        return None

    async def generate_batch_questions(
        self,
        batch_config_id: str,
        seed_question_id: Optional[str] = None
    ) -> List[CandidateQuestion]:
        """Generate multiple questions using a batch config"""

        print(f"\n🔄 Starting batch generation: {batch_config_id}")

        # Get batch configuration
        batch_config = self.config_manager.batch_configs.get(batch_config_id)
        if not batch_config:
            print(f"❌ Batch config '{batch_config_id}' not found")
            return []

        print(f"   Description: {batch_config.description}")
        print(f"   Total questions: {batch_config.total_questions}")
        print(f"   Configs: {', '.join(batch_config.configs_to_use)}")

        # Create generation configs
        generation_configs = self.config_manager.create_batch_generation_configs(
            batch_config_id=batch_config_id,
            seed_question_id=seed_question_id
        )

        if not generation_configs:
            print(f"❌ No generation configs created for batch {batch_config_id}")
            return []

        # Generate questions
        generated_questions = []
        total_configs = len(generation_configs)

        for i, generation_config in enumerate(generation_configs, 1):
            print(f"\n📝 Question {i}/{total_configs}")

            model = self._create_model(generation_config.llm_model_generation)
            agent = QuestionGeneratorAgent(model=model, db_client=self.db_client, debug=False)

            start_time = time.time()
            candidate_question = await agent.generate_question(generation_config)
            generation_time = time.time() - start_time

            if candidate_question:
                success = await self.db_client.save_candidate_question(candidate_question)
                if success:
                    generated_questions.append(candidate_question)
                    print(f"   ✅ Generated and saved ({generation_time:.2f}s)")
                else:
                    print(f"   ❌ Generated but failed to save")
            else:
                print(f"   ❌ Generation failed")

            # Small delay to avoid rate limits
            if i < total_configs:
                await asyncio.sleep(1)

        print(f"\n🎯 Batch generation complete: {len(generated_questions)}/{total_configs} questions generated")
        return generated_questions

    async def generate_questions_for_config(
        self,
        config_id: str,
        count: int = 1,
        target_grade: Optional[int] = None,
        seed_question_id: Optional[str] = None
    ) -> List[CandidateQuestion]:
        """Generate multiple questions using a single config"""

        print(f"\n🔄 Generating {count} questions with config: {config_id}")

        generated_questions = []

        for i in range(count):
            print(f"\n📝 Question {i+1}/{count}")
            question = await self.generate_single_question(
                config_id=config_id,
                target_grade=target_grade,
                seed_question_id=seed_question_id
            )

            if question:
                generated_questions.append(question)

            # Small delay between generations
            if i < count - 1:
                await asyncio.sleep(1)

        print(f"\n🎯 Generation complete: {len(generated_questions)}/{count} questions generated")
        return generated_questions

    def list_configs(self):
        """List all available configurations"""
        print("\n📋 Available Configurations:")
        print("\n🔧 Individual Configs:")
        for config_id in self.config_manager.list_available_configs():
            info = self.config_manager.get_config_info(config_id)
            if info:
                print(f"   {config_id}: {info['description']}")
                print(f"      Grades: {info['target_grades']}")
                print(f"      Model: {info['models']['generation']}")
                print(f"      Marks: {info['marks']}")

        print("\n📦 Batch Configs:")
        for batch_id, batch_config in self.config_manager.batch_configs.items():
            print(f"   {batch_id}: {batch_config.description}")
            print(f"      Total questions: {batch_config.total_questions}")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Generate IGCSE Mathematics candidate questions")

    # Mutually exclusive group for config selection
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--config", help="Single config ID to use")
    config_group.add_argument("--batch", help="Batch config ID to use")
    config_group.add_argument("--list", action="store_true", help="List available configurations")

    # Optional parameters
    parser.add_argument("--grade", type=int, choices=range(1, 10), help="Target grade (1-9)")
    parser.add_argument("--count", type=int, default=1, help="Number of questions to generate (for single config)")
    parser.add_argument("--seed", help="Seed question ID for variation")

    args = parser.parse_args()

    generator = CandidateQuestionGenerator()

    try:
        if args.list:
            generator.list_configs()
            return

        await generator.setup()

        if args.config:
            await generator.generate_questions_for_config(
                config_id=args.config,
                count=args.count,
                target_grade=args.grade,
                seed_question_id=args.seed
            )
        elif args.batch:
            await generator.generate_batch_questions(
                batch_config_id=args.batch,
                seed_question_id=args.seed
            )

    except KeyboardInterrupt:
        print("\n⏹️  Generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await generator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
