#!/usr/bin/env python3
"""
Test Complete Workflow - Configurable Sample Size

Test the complete Generation → Review → Refine → Auto-publish workflow
using fast reliable models: GPT-4o and Gemini 2.5 Flash.
No database cleanup, configurable sample size, shows distribution and samples.
"""

import asyncio
import sys
import os
import time
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.generate_300_comprehensive import ComprehensiveQuestionGenerator


async def test_complete_workflow(sample_size: int = 5, auto_publish: bool = True):
    """Test the complete workflow with configurable sample size using fast models"""
    print(f"🧪 Testing Complete Workflow ({sample_size} questions)")
    print("🚀 Using fast models: GPT-4o, Gemini 2.5 Flash")
    print(f"🗄️  Database: NO cleanup (preserving existing data)")
    print(f"📡 Auto-publish: {'✅ ENABLED' if auto_publish else '❌ DISABLED'}")
    print("="*60)

    start_time = time.time()

    generator = ComprehensiveQuestionGenerator(
        auto_publish=auto_publish,  # Enable auto-publish for testing
        thinking_models_enabled=False  # Disable slow thinking models
    )

    # Override model configs to use only fast reliable models
    generator.model_configs = [
        {'name': 'gpt-4o', 'type': 'openai', 'thinking': False, 'weight': 60},
        {'name': 'gpt-4o-mini', 'type': 'openai', 'thinking': False, 'weight': 40},
        # Note: Gemini 2.5 Flash will be added when available in the service
    ]

    print(f"✅ Model configurations:")
    for config in generator.model_configs:
        print(f"   💡 {config['name']} (weight: {config['weight']}%)")

    try:
        # NO database cleanup - preserve existing data
        print(f"\n🗄️  Skipping database cleanup (preserving existing questions)")

        # Setup workflow
        workflow = await generator.setup_workflow()

        # Generate questions
        print(f"\n🔄 Generating {sample_size} test questions...")
        batch_start = time.time()
        results = await generator.generate_question_batch(workflow, batch_size=sample_size)
        batch_time = time.time() - batch_start

        # Analyze results
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]

        print(f"\n📊 GENERATION RESULTS:")
        print(f"   ✅ Successful: {len(successful)}/{sample_size}")
        print(f"   ❌ Failed: {len(failed)}/{sample_size}")
        print(f"   ⏱️  Total time: {batch_time:.1f}s")
        print(f"   ⚡ Avg per question: {batch_time/sample_size:.1f}s")

        # Show distribution analysis
        if successful:
            print(f"\n📈 DISTRIBUTION ANALYSIS:")

            # Model distribution
            model_counts = {}
            decision_counts = {}
            grade_counts = {}

            for result in successful:
                model = result.get('model_used', 'unknown')
                decision = str(result.get('final_decision', 'unknown'))
                grade = result.get('target_grade', 'unknown')

                model_counts[model] = model_counts.get(model, 0) + 1
                decision_counts[decision] = decision_counts.get(decision, 0) + 1
                grade_counts[grade] = grade_counts.get(grade, 0) + 1

            print(f"   🤖 Model Distribution:")
            for model, count in sorted(model_counts.items()):
                percentage = (count / len(successful)) * 100
                print(f"      {model}: {count} ({percentage:.1f}%)")

            print(f"   🚦 Quality Decision Distribution:")
            for decision, count in sorted(decision_counts.items()):
                percentage = (count / len(successful)) * 100
                print(f"      {decision}: {count} ({percentage:.1f}%)")

            print(f"   📚 Grade Distribution:")
            for grade, count in sorted(grade_counts.items()):
                percentage = (count / len(successful)) * 100
                print(f"      Grade {grade}: {count} ({percentage:.1f}%)")

            # Show sample questions for inspection
            print(f"\n🔍 SAMPLE QUESTIONS FOR INSPECTION:")
            print("="*80)

            for i, result in enumerate(successful[:min(3, len(successful))], 1):
                print(f"\n📝 SAMPLE {i}:")
                print(f"   🤖 Model: {result.get('model_used', 'unknown')}")
                print(f"   📊 Grade: {result.get('target_grade', 'unknown')}")
                print(f"   🏆 Marks: {result.get('marks', 'unknown')}")
                print(f"   🚦 Decision: {result.get('final_decision', 'unknown')}")
                print(f"   ⭐ Score: {result.get('review_score', 0.0):.2f}")
                print(f"   ⏱️  Time: {result.get('processing_time', 0.0):.1f}s")
                print(f"   📡 Auto-published: {result.get('auto_published', False)}")

                # Get question details if available
                question_id = result.get('question_id', 'unknown')
                print(f"   🆔 Question ID: {question_id}")
                print(f"   📋 Subject refs: {result.get('subject_refs', [])}")

                print("   " + "-"*70)

        if failed:
            print(f"\n❌ FAILED QUESTIONS:")
            for i, result in enumerate(failed, 1):
                error = result.get('error', 'Unknown error')
                config = result.get('config', {})
                model_name = config.get('model_config', {}).get('name', 'unknown') if config else 'unknown'
                print(f"   {i}. {model_name}: {error}")

        # Final assessment
        success_rate = len(successful) / sample_size * 100
        total_time = time.time() - start_time

        print(f"\n🎯 FINAL ASSESSMENT:")
        print(f"   📊 Success Rate: {success_rate:.1f}%")
        print(f"   ⏱️  Total Runtime: {total_time:.1f}s")
        print(f"   ⚡ Questions/minute: {(len(successful) / total_time * 60):.1f}")

        if auto_publish:
            auto_published_count = sum(1 for r in successful if r.get('auto_published', False))
            print(f"   📡 Auto-published: {auto_published_count}/{len(successful)}")

        threshold = 70  # 70% success rate minimum
        if success_rate >= threshold:
            print(f"\n✅ WORKFLOW TEST PASSED! ({len(successful)}/{sample_size} questions generated)")
            if auto_publish and auto_published_count > 0:
                print(f"✅ AUTO-PUBLISH WORKING! ({auto_published_count} questions published)")
            return True
        else:
            print(f"\n❌ WORKFLOW TEST FAILED! Only {len(successful)}/{sample_size} questions generated")
            print(f"   Minimum required: {threshold}% success rate")
            return False

    except Exception as e:
        print(f"\n❌ WORKFLOW TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """CLI interface for workflow testing"""
    parser = argparse.ArgumentParser(description="Test complete question generation workflow")
    parser.add_argument("--sample-size", "-n", type=int, default=5,
                        help="Number of questions to generate (default: 5)")
    parser.add_argument("--auto-publish", action="store_true", default=False,
                        help="Enable auto-publishing to Payload CMS")
    parser.add_argument("--no-auto-publish", action="store_false", dest="auto_publish",
                        help="Disable auto-publishing (default)")

    args = parser.parse_args()

    print(f"Starting workflow test with {args.sample_size} questions...")
    success = asyncio.run(test_complete_workflow(args.sample_size, args.auto_publish))
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
