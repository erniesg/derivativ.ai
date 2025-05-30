import click
import asyncio
import json
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.generation_service import QuestionGenerationService
from models.generation_config import GenerationConfig

@click.group()
@click.option('--debug', is_flag=True, help='Enable debug output')
@click.pass_context
def cli(ctx, debug):
    """IGCSE Mathematics Question Generation System"""
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug

@cli.command()
@click.option('--model', default='gpt-4o-mini', help='LLM model to use')
@click.option('--provider', default='openai', help='LLM provider (openai, claude, gemini, huggingface)')
@click.option('--grade', type=int, default=6, help='Target grade (1-9)')
@click.option('--topic', help='Specific topic to focus on')
@click.option('--seed-question', help='Seed question ID for inspiration')
@click.option('--use-local', is_flag=True, help='Use local 2025p1.json file instead of database')
@click.option('--no-assets', is_flag=True, help='Only use questions without diagrams/assets')
@click.option('--full-set', is_flag=True, help='If using seed question, get the full question set (all parts)')
@click.option('--output', help='Output file path (optional)')
@click.pass_context
def generate(ctx, model, provider, grade, topic, seed_question, use_local, no_assets, full_set, output):
    """Generate a new IGCSE Mathematics question"""
    debug = ctx.obj.get('debug', False)

    # Enhanced help examples in debug mode
    if debug:
        click.echo("=== GENERATION EXAMPLES ===")
        click.echo("Basic generation: python -m src.cli generate --grade 7 --topic 'Algebra'")
        click.echo("With seed (database): python -m src.cli generate --seed-question '0580_SP_25_P1_q1a'")
        click.echo("With local file: python -m src.cli generate --use-local --no-assets")
        click.echo("Full question set: python -m src.cli generate --seed-question 'Q1a' --use-local --full-set")
        click.echo("Debug mode: python -m src.cli generate --debug --use-local")
        click.echo("=" * 30)

    asyncio.run(_generate_question(model, provider, grade, topic, seed_question, use_local, no_assets, full_set, output, debug))

@cli.command()
@click.option('--use-local', is_flag=True, help='Use local 2025p1.json file instead of database')
@click.option('--no-assets', is_flag=True, help='Only show questions without diagrams/assets')
@click.option('--limit', type=int, default=10, help='Number of questions to show')
@click.option('--command-word', help='Filter by command word')
@click.option('--min-marks', type=int, help='Minimum marks')
@click.option('--max-marks', type=int, help='Maximum marks')
@click.pass_context
def browse(ctx, use_local, no_assets, limit, command_word, min_marks, max_marks):
    """Browse available questions for testing and selection"""
    debug = ctx.obj.get('debug', False)
    asyncio.run(_browse_questions(use_local, no_assets, limit, command_word, min_marks, max_marks, debug))

async def _generate_question(model, provider, grade, topic, seed_question, use_local, no_assets, full_set, output, debug):
    """Async wrapper for question generation"""
    try:
        # Initialize service with debug flag
        service = QuestionGenerationService(debug=debug)

        # Handle seed question logic
        if seed_question and use_local:
            if debug:
                click.echo(f"[DEBUG] Using local file mode with seed: {seed_question}")

            if full_set:
                # Get full question set from local file
                questions = await service.db_client.get_question_set_from_local_file(seed_question)
                if questions:
                    click.echo(f"Found {len(questions)} parts in question set:")
                    for q in questions:
                        click.echo(f"  - {q.get('question_id_local', 'N/A')}: {q.get('raw_text_content', '')[:60]}...")
                    seed_question = questions[0].get('question_id_global', seed_question)
                else:
                    click.echo(f"No question set found for {seed_question}, using as single question")

        elif not seed_question and (no_assets or use_local):
            # Auto-select a seed question without assets
            if use_local:
                questions = await service.db_client.get_questions_from_local_file(
                    limit=1,
                    exclude_with_assets=no_assets
                )
            else:
                questions = await service.db_client.get_questions_without_assets(limit=1)

            if questions:
                seed_question = questions[0].get('question_id_global')
                if debug:
                    click.echo(f"[DEBUG] Auto-selected seed question: {seed_question}")
            else:
                if debug:
                    click.echo("[DEBUG] No suitable questions found, proceeding without seed")

        # Configure generation
        config = GenerationConfig(
            model=model,
            provider=provider,
            target_grade=grade,
            topic_focus=topic,
            seed_question_id=seed_question
        )

        if debug:
            click.echo(f"[DEBUG] Generation config: {config}")

        # Generate question
        click.echo("Generating question...")
        result = await service.generate_question(config)

        if result:
            click.echo("✅ Question generated successfully!")
            click.echo("\n" + "="*50)
            click.echo(f"Question ID: {result.question_id_global}")
            click.echo(f"Grade: {result.target_grade_input}")
            click.echo(f"Command Word: {result.command_word.value}")
            click.echo(f"Marks: {result.marks}")
            click.echo("\nQuestion:")
            click.echo(result.raw_text_content)

            # Show solution summary
            final_answers = result.solution_and_marking_scheme.final_answers_summary
            if final_answers:
                click.echo(f"\nAnswer: {final_answers[0].get('answer_value', 'N/A')}")

            # Save to file if requested
            if output:
                with open(output, 'w') as f:
                    json.dump(result.model_dump(), f, indent=2, default=str)
                click.echo(f"\n💾 Saved to {output}")
        else:
            click.echo("❌ Failed to generate question")

    except Exception as e:
        click.echo(f"❌ Error: {e}")
        if debug:
            import traceback
            click.echo(traceback.format_exc())

async def _browse_questions(use_local, no_assets, limit, command_word, min_marks, max_marks, debug):
    """Async wrapper for browsing questions"""
    try:
        service = QuestionGenerationService(debug=debug)

        if use_local:
            if debug:
                click.echo("[DEBUG] Using local file mode")
            questions = await service.db_client.get_questions_from_local_file(
                limit=limit * 2,  # Get more for filtering
                exclude_with_assets=no_assets
            )
        else:
            if debug:
                click.echo("[DEBUG] Using database mode")
            questions = await service.db_client.get_questions_without_assets(
                limit=limit * 2,
                command_words=[command_word] if command_word else None
            )

        # Apply additional filters
        if command_word:
            questions = [q for q in questions if q.get('command_word', '').lower() == command_word.lower()]

        if min_marks:
            questions = [q for q in questions if q.get('marks', 0) >= min_marks]

        if max_marks:
            questions = [q for q in questions if q.get('marks', 0) <= max_marks]

        # Limit final results
        questions = questions[:limit]

        if not questions:
            click.echo("❌ No questions found matching criteria")
            return

        click.echo(f"📚 Found {len(questions)} questions:")
        click.echo("=" * 80)

        for i, q in enumerate(questions, 1):
            taxonomy = q.get('taxonomy', {})
            topic_path = taxonomy.get('topic_path', [])
            if isinstance(topic_path, list):
                topic_str = ' > '.join(topic_path)
            else:
                topic_str = str(topic_path)

            click.echo(f"{i:2d}. {q.get('question_id_global', 'N/A'):20s} | {q.get('command_word', 'N/A'):12s} | {q.get('marks', 0):2d} marks | {topic_str}")

            # Show truncated question text
            text = q.get('raw_text_content', '')
            if len(text) > 60:
                text = text[:60] + "..."
            click.echo(f"     {text}")

            # Show if it has assets
            asset_count = q.get('asset_count', 0) or (1 if q.get('assets') else 0)
            if asset_count > 0:
                click.echo(f"     🖼️  Has {asset_count} asset(s)")

            click.echo()

        # Show usage examples
        if questions:
            example_id = questions[0].get('question_id_global', 'Q1a')
            click.echo("💡 Usage examples:")
            click.echo(f"   Generate with seed: python -m src.cli generate --seed-question '{example_id}' {'--use-local' if use_local else ''}")
            click.echo(f"   Get full set:       python -m src.cli generate --seed-question '{example_id}' --use-local --full-set")

    except Exception as e:
        click.echo(f"❌ Error: {e}")
        if debug:
            import traceback
            click.echo(traceback.format_exc())

if __name__ == '__main__':
    cli()
