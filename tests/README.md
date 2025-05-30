# Tests Directory

This directory contains all test files for the IGCSE Mathematics Question Generation System.

## Test Files

### Core Generation Tests
- `test_generation.py` - End-to-end generation tests with database
- `test_generation_local.py` - Generation tests using local file data
- `test_llm_only.py` - Direct LLM testing without full pipeline

### Component Tests
- `test_json_parsing.py` - JSON parsing and validation tests
- `test_local_questions.py` - Local question file handling tests
- `test_smart_question_sets.py` - Question set detection and grouping tests

### Model Tests
- `test_openai_server_model.py` - OpenAI model integration tests
- `test_hf_inference_model.py` - Hugging Face model integration tests

## Running Tests

### Individual Tests
```bash
# Run a specific test
python tests/test_generation.py

# Run with debug output
python tests/test_generation.py --debug
```

### Using the CLI for Testing
```bash
# Test basic generation
python -m src.cli generate --grade 6 --use-local --debug

# Browse available questions
python -m src.cli browse --use-local --limit 5
```

## Test Environment Setup

1. **For database tests**: Ensure `.env` file has `NEON_DATABASE_URL` set
2. **For local tests**: Ensure `data/2025p1.json` exists
3. **For LLM tests**: Set appropriate API keys in `.env`:
   - `OPENAI_API_KEY` for OpenAI models
   - `ANTHROPIC_API_KEY` for Claude
   - `GOOGLE_API_KEY` for Gemini
   - `HF_TOKEN` for Hugging Face models

## Import Structure

All tests now use the clean import structure:

```python
# Clean imports from main package
from src import QuestionGenerationService, GenerationConfig, CommandWord

# Specific model imports
from src.models import QuestionTaxonomy, SolverAlgorithm
```

## Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Multi-component pipeline testing
- **E2E Tests**: Full system testing with real data
- **Performance Tests**: Generation speed and quality testing
