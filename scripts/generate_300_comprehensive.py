#!/usr/bin/env python3
"""
Comprehensive 300 Question Generation Script

This script demonstrates the complete end-to-end workflow:
1. Clear existing database tables
2. Generate 300 questions with random sampling across:
   - Grades (1-9)
   - Subject content references
   - Skill tags
   - Models (thinking vs regular)
3. Use QualityControlWorkflow for Generation → Review → Refine → Auto-publish
4. Save complete audit trails (LLM interactions, sessions, review results)
5. Generate coverage/distribution analysis

Features:
- Uses DatabaseManager for complete audit trails
- Random sampling from enums for diversity
- Auto-publish integration (configurable)
- Progress tracking and error handling
- Comprehensive statistics and coverage analysis
"""

import asyncio
import os
import sys
import random
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import uuid

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.services.quality_control_workflow import QualityControlWorkflow, QualityDecision
from src.services.database_manager import DatabaseManager
from src.services.config_manager import ConfigManager
from src.agents import QuestionGeneratorAgent, ReviewAgent, RefinementAgent, MarkerAgent
from src.models import GenerationConfig, LLMModel, CalculatorPolicy, CommandWord
from src.models.enums import get_valid_subject_refs, get_valid_skill_tags
from src.database import NeonDBClient
from smolagents import OpenAIServerModel, LiteLLMModel, AmazonBedrockServerModel, InferenceClientModel

from dotenv import load_dotenv
load_dotenv()


class ComprehensiveQuestionGenerator:
    """Generates 300 questions using complete quality control workflow"""

    def __init__(self, auto_publish: bool = False, thinking_models_enabled: bool = True):
        self.auto_publish = auto_publish
        self.thinking_models_enabled = thinking_models_enabled

        # Initialize database manager (uses complete schema)
        database_url = os.getenv("NEON_DATABASE_URL")
        if not database_url:
            raise ValueError("NEON_DATABASE_URL environment variable required")

        self.database_manager = DatabaseManager(database_url)
        self.config_manager = ConfigManager()

        # Get enum values for sampling
        self.subject_refs = get_valid_subject_refs()
        self.skill_tags = get_valid_skill_tags()

        # Model configurations for diversity
        self.model_configs = self._setup_model_configs()

        # Statistics tracking
        self.stats = {
            'total_requested': 300,
            'total_generated': 0,
            'auto_approved': 0,
            'manual_review': 0,
            'refined': 0,
            'regenerated': 0,
            'rejected': 0,
            'auto_published': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None,
            'by_grade': {},
            'by_model': {},
            'by_subject_area': {},
            'by_skill_tags': {},
            'by_quality_decision': {},
            'review_scores': [],
            'processing_times': [],
            'error_details': []
        }

    def _setup_model_configs(self) -> List[Dict[str, Any]]:
        """Setup diverse model configurations for testing"""
        configs = []

        # OpenAI models (reliable and cost-effective)
        configs.extend([
            {'name': 'gpt-4o-mini', 'type': 'openai', 'thinking': False, 'weight': 40},
            {'name': 'gpt-4o', 'type': 'openai', 'thinking': False, 'weight': 30},
        ])

        # HuggingFace thinking models (if enabled)
        if self.thinking_models_enabled:
            configs.extend([
                {'name': 'Qwen/Qwen3-235B-A22B', 'type': 'hf', 'thinking': True, 'weight': 20},
                {'name': 'deepseek-ai/DeepSeek-R1-0528', 'type': 'hf', 'thinking': True, 'weight': 10},
            ])

        return configs

    def _create_model(self, config: Dict[str, Any]):
        """Create model instance based on configuration"""
        model_name = config['name']
        model_type = config['type']

        try:
            if model_type == 'openai':
                return OpenAIServerModel(
                    model_id=model_name,
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            elif model_type == 'bedrock':
                return AmazonBedrockServerModel(
                    model_id=model_name,
                    client_kwargs={'region_name': os.getenv("AWS_REGION", "us-east-1")}
                )
            elif model_type == 'hf':
                return InferenceClientModel(
                    model_id=model_name,
                    provider="auto",
                    token=os.getenv("HF_TOKEN"),
                    max_tokens=8000
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except Exception as e:
            print(f"⚠️ Failed to create model {model_name}: {e}")
            # Fallback to GPT-4o-mini
            return OpenAIServerModel(
                model_id="gpt-4o-mini",
                api_key=os.getenv("OPENAI_API_KEY")
            )

    def _sample_random_config(self) -> Dict[str, Any]:
        """Sample random configuration for diversity"""

        # Sample model (weighted by preference)
        weights = [config['weight'] for config in self.model_configs]
        model_config = random.choices(self.model_configs, weights=weights)[0]

        # Sample educational parameters
        target_grade = random.randint(1, 9)
        desired_marks = random.choices([1, 2, 3, 4, 5], weights=[10, 30, 35, 20, 5])[0]

        # Sample subject content references (2-4 refs)
        num_refs = random.randint(2, 4)
        subject_content_refs = random.sample(self.subject_refs, num_refs)

        # Sample calculator policy
        calculator_policy = random.choice(list(CalculatorPolicy))

        # Sample command word
        command_word = random.choice(list(CommandWord))

        return {
            'model_config': model_config,
            'target_grade': target_grade,
            'desired_marks': desired_marks,
            'subject_content_references': subject_content_refs,
            'calculator_policy': calculator_policy,
            'command_word_override': command_word,
            'temperature': random.uniform(0.6, 0.8),
            'max_tokens': random.randint(3000, 5000)
        }

    async def clear_database_tables(self):
        """Clear relevant database tables for fresh start"""
        print("🗑️  Clearing database tables for fresh start...")

        # Connect directly to clear tables
        import asyncpg
        conn = await asyncpg.connect(os.getenv("NEON_DATABASE_URL"))

        try:
            # Clear in reverse dependency order
            tables_to_clear = [
                'deriv_manual_review_queue',
                'deriv_error_logs',
                'deriv_review_results',
                'deriv_candidate_questions',
                'deriv_llm_interactions',
                'deriv_generation_sessions'
            ]

            for table in tables_to_clear:
                try:
                    result = await conn.execute(f"DELETE FROM {table}")
                    row_count = int(result.split()[-1]) if result.split()[-1].isdigit() else 0
                    print(f"   ✅ Cleared {table}: {row_count} rows deleted")
                except Exception as e:
                    print(f"   ⚠️ Could not clear {table}: {e}")

            print("✅ Database tables cleared successfully!")

        finally:
            await conn.close()

    async def setup_workflow(self) -> QualityControlWorkflow:
        """Setup the complete quality control workflow"""
        print("🔧 Setting up quality control workflow...")

        # Initialize database manager
        await self.database_manager.initialize()

        # Create model instances (using GPT-4o-mini for agents, will vary for generation)
        base_model = OpenAIServerModel(
            model_id="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Create agents (these will be overridden with different models during generation)
        generator_agent = QuestionGeneratorAgent(base_model, None, debug=False)
        review_agent = ReviewAgent(base_model, None, debug=False)
        marker_agent = MarkerAgent(base_model, None, debug=False)
        refinement_agent = RefinementAgent(base_model, self.config_manager)

        # Create quality control workflow
        workflow = QualityControlWorkflow(
            review_agent=review_agent,
            refinement_agent=refinement_agent,
            generator_agent=generator_agent,
            database_manager=self.database_manager,
            quality_thresholds={
                'auto_approve': 0.85,
                'manual_review': 0.70,
                'refine': 0.60,
                'regenerate': 0.40
            },
            auto_publish=self.auto_publish
        )

        print(f"✅ Quality control workflow setup complete!")
        print(f"   Auto-publish: {'✅ Enabled' if self.auto_publish else '❌ Disabled'}")
        print(f"   Thinking models: {'✅ Enabled' if self.thinking_models_enabled else '❌ Disabled'}")

        return workflow

    async def generate_question_batch(self, workflow: QualityControlWorkflow, batch_size: int = 10) -> List[Dict[str, Any]]:
        """Generate a batch of questions with diverse configurations"""
        results = []

        for i in range(batch_size):
            try:
                start_time = time.time()

                # Sample random configuration
                config = self._sample_random_config()

                # Create GenerationConfig
                generation_config = GenerationConfig(
                    generation_id=uuid.uuid4(),
                    target_grade=config['target_grade'],
                    calculator_policy=config['calculator_policy'],
                    desired_marks=config['desired_marks'],
                    subject_content_references=config['subject_content_references'],
                    command_word_override=config['command_word_override'],
                    llm_model_generation=LLMModel(config['model_config']['name']),
                    llm_model_marking_scheme=LLMModel("gpt-4o-mini"),
                    llm_model_review=LLMModel("gpt-4o-mini"),
                    temperature=config['temperature'],
                    max_tokens=config['max_tokens'],
                    prompt_template_version_generation="v1.3"
                )

                # Override generator with specific model for this question
                model = self._create_model(config['model_config'])
                workflow.generator_agent = QuestionGeneratorAgent(model, None, debug=False)

                # Generate initial question
                question = await workflow.generator_agent.generate_question(generation_config)

                if not question:
                    raise Exception("Question generation failed")

                # Process through quality control workflow
                session_id = str(uuid.uuid4())
                workflow_result = await workflow.process_question(
                    question, session_id, generation_config.__dict__
                )

                processing_time = time.time() - start_time

                # Record results
                result = {
                    'success': workflow_result['success'],
                    'question_id': question.question_id_local,
                    'session_id': session_id,
                    'model_used': config['model_config']['name'],
                    'target_grade': config['target_grade'],
                    'marks': config['desired_marks'],
                    'subject_refs': config['subject_content_references'],
                    'final_decision': workflow_result.get('final_decision'),
                    'review_score': workflow_result.get('review_score', 0.0),
                    'processing_time': processing_time,
                    'auto_published': workflow_result.get('payload_published', False),
                    'iterations': workflow_result.get('total_iterations', 1),
                    'config': config
                }

                results.append(result)
                self._update_stats(result)

                print(f"   📝 Question {i+1}/{batch_size}: {result['final_decision']} "
                      f"(score: {result['review_score']:.2f}, {processing_time:.1f}s)")

            except Exception as e:
                error_result = {
                    'success': False,
                    'error': str(e),
                    'config': config if 'config' in locals() else None
                }
                results.append(error_result)
                self.stats['errors'] += 1
                self.stats['error_details'].append(str(e))
                print(f"   ❌ Question {i+1}/{batch_size}: ERROR - {e}")

        return results

    def _update_stats(self, result: Dict[str, Any]):
        """Update comprehensive statistics"""
        if not result['success']:
            return

        self.stats['total_generated'] += 1

        # Decision tracking
        decision = result.get('final_decision')
        if decision:
            if hasattr(decision, 'value'):
                decision = decision.value
            self.stats['by_quality_decision'][decision] = self.stats['by_quality_decision'].get(decision, 0) + 1

            if decision == 'auto_approve':
                self.stats['auto_approved'] += 1
            elif decision == 'manual_review':
                self.stats['manual_review'] += 1
            elif decision == 'refine':
                self.stats['refined'] += 1
            elif decision == 'regenerate':
                self.stats['regenerated'] += 1
            elif decision == 'reject':
                self.stats['rejected'] += 1

        # Auto-publish tracking
        if result.get('auto_published'):
            self.stats['auto_published'] += 1

        # Grade distribution
        grade = result['target_grade']
        self.stats['by_grade'][grade] = self.stats['by_grade'].get(grade, 0) + 1

        # Model distribution
        model = result['model_used']
        self.stats['by_model'][model] = self.stats['by_model'].get(model, 0) + 1

        # Subject area distribution
        for ref in result['subject_refs']:
            self.stats['by_subject_area'][ref] = self.stats['by_subject_area'].get(ref, 0) + 1

        # Performance tracking
        self.stats['review_scores'].append(result.get('review_score', 0.0))
        self.stats['processing_times'].append(result['processing_time'])

    async def generate_300_questions(self):
        """Generate 300 questions using the complete workflow"""
        print("🚀 Starting comprehensive 300 question generation...")
        print(f"   Target: {self.stats['total_requested']} questions")
        print(f"   Auto-publish: {'✅ Enabled' if self.auto_publish else '❌ Disabled'}")
        print(f"   Thinking models: {'✅ Enabled' if self.thinking_models_enabled else '❌ Disabled'}")

        self.stats['start_time'] = datetime.utcnow()

        # Clear database
        await self.clear_database_tables()

        # Setup workflow
        workflow = await self.setup_workflow()

        # Generate in batches for better progress tracking
        batch_size = 10
        total_batches = (self.stats['total_requested'] + batch_size - 1) // batch_size

        all_results = []

        for batch_num in range(total_batches):
            questions_remaining = self.stats['total_requested'] - len(all_results)
            current_batch_size = min(batch_size, questions_remaining)

            print(f"\n📦 Batch {batch_num + 1}/{total_batches} ({current_batch_size} questions)")

            batch_results = await self.generate_question_batch(workflow, current_batch_size)
            all_results.extend(batch_results)

            # Progress update
            success_count = sum(1 for r in batch_results if r.get('success', False))
            print(f"   ✅ Batch complete: {success_count}/{current_batch_size} successful")

            # Brief pause between batches
            await asyncio.sleep(1)

        self.stats['end_time'] = datetime.utcnow()

        # Generate final report
        await self.generate_final_report()

        return all_results

    async def generate_final_report(self):
        """Generate comprehensive final report with coverage analysis"""
        total_time = (self.stats['end_time'] - self.stats['start_time']).total_seconds()

        print("\n" + "="*60)
        print("📊 COMPREHENSIVE GENERATION REPORT")
        print("="*60)

        # Summary Statistics
        print(f"\n🎯 SUMMARY STATISTICS:")
        print(f"   Total Requested: {self.stats['total_requested']}")
        print(f"   Total Generated: {self.stats['total_generated']}")
        print(f"   Success Rate: {(self.stats['total_generated']/self.stats['total_requested']*100):.1f}%")
        print(f"   Total Time: {total_time:.1f}s")
        print(f"   Average per Question: {total_time/max(self.stats['total_generated'], 1):.1f}s")
        print(f"   Errors: {self.stats['errors']}")

        # Quality Control Results
        print(f"\n🚦 QUALITY CONTROL DECISIONS:")
        print(f"   Auto-Approved: {self.stats['auto_approved']} ({(self.stats['auto_approved']/max(self.stats['total_generated'], 1)*100):.1f}%)")
        print(f"   Manual Review: {self.stats['manual_review']} ({(self.stats['manual_review']/max(self.stats['total_generated'], 1)*100):.1f}%)")
        print(f"   Refined: {self.stats['refined']} ({(self.stats['refined']/max(self.stats['total_generated'], 1)*100):.1f}%)")
        print(f"   Regenerated: {self.stats['regenerated']} ({(self.stats['regenerated']/max(self.stats['total_generated'], 1)*100):.1f}%)")
        print(f"   Rejected: {self.stats['rejected']} ({(self.stats['rejected']/max(self.stats['total_generated'], 1)*100):.1f}%)")

        if self.auto_publish:
            print(f"   Auto-Published: {self.stats['auto_published']} ({(self.stats['auto_published']/max(self.stats['auto_approved'], 1)*100):.1f}% of approved)")

        # Grade Distribution
        print(f"\n📚 GRADE DISTRIBUTION:")
        for grade in sorted(self.stats['by_grade'].keys()):
            count = self.stats['by_grade'][grade]
            percentage = (count/max(self.stats['total_generated'], 1)*100)
            print(f"   Grade {grade}: {count} questions ({percentage:.1f}%)")

        # Model Distribution
        print(f"\n🤖 MODEL DISTRIBUTION:")
        for model, count in sorted(self.stats['by_model'].items()):
            percentage = (count/max(self.stats['total_generated'], 1)*100)
            thinking = "🧠" if any(model.startswith(prefix) for prefix in ['Qwen', 'deepseek']) else "💡"
            print(f"   {thinking} {model}: {count} questions ({percentage:.1f}%)")

        # Subject Coverage
        print(f"\n📖 SUBJECT CONTENT COVERAGE:")
        unique_refs = len(set(self.stats['by_subject_area'].keys()))
        total_refs = len(self.subject_refs)
        coverage = (unique_refs/total_refs*100)
        print(f"   Coverage: {unique_refs}/{total_refs} subject references ({coverage:.1f}%)")

        # Top subject areas
        top_subjects = sorted(self.stats['by_subject_area'].items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"   Top 10 Subject Areas:")
        for ref, count in top_subjects:
            print(f"      {ref}: {count} questions")

        # Performance Metrics
        if self.stats['review_scores']:
            avg_score = sum(self.stats['review_scores']) / len(self.stats['review_scores'])
            min_score = min(self.stats['review_scores'])
            max_score = max(self.stats['review_scores'])
            print(f"\n⭐ QUALITY METRICS:")
            print(f"   Average Review Score: {avg_score:.3f}")
            print(f"   Score Range: {min_score:.3f} - {max_score:.3f}")

        if self.stats['processing_times']:
            avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            min_time = min(self.stats['processing_times'])
            max_time = max(self.stats['processing_times'])
            print(f"\n⏱️  PERFORMANCE METRICS:")
            print(f"   Average Processing Time: {avg_time:.1f}s")
            print(f"   Time Range: {min_time:.1f}s - {max_time:.1f}s")

        # Database Statistics
        await self._show_database_statistics()

        print("\n" + "="*60)
        print("✅ GENERATION COMPLETE")
        print("="*60)

    async def _show_database_statistics(self):
        """Show database statistics after generation"""
        print(f"\n💾 DATABASE STATISTICS:")

        import asyncpg
        conn = await asyncpg.connect(os.getenv("NEON_DATABASE_URL"))

        try:
            # Count records in each table
            tables = [
                'deriv_generation_sessions',
                'deriv_llm_interactions',
                'deriv_candidate_questions',
                'deriv_review_results',
                'deriv_error_logs'
            ]

            for table in tables:
                try:
                    count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                    print(f"   {table}: {count} records")
                except Exception as e:
                    print(f"   {table}: Error counting - {e}")

        finally:
            await conn.close()


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate 300 comprehensive questions with quality control")
    parser.add_argument("--auto-publish", action="store_true", help="Enable auto-publishing to Payload CMS")
    parser.add_argument("--no-thinking-models", action="store_true", help="Disable thinking models (Qwen, DeepSeek)")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for generation")

    args = parser.parse_args()

    try:
        generator = ComprehensiveQuestionGenerator(
            auto_publish=args.auto_publish,
            thinking_models_enabled=not args.no_thinking_models
        )

        results = await generator.generate_300_questions()

        print(f"\n🎉 Generation complete! Check database for full audit trails.")
        if args.auto_publish:
            print(f"📡 Auto-published questions should be visible in Payload CMS.")

    except KeyboardInterrupt:
        print("\n⏹️  Generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
