# IGCSE Mathematics Question Generation System

An AI-powered pipeline for generating IGCSE Mathematics (0580) practice questions using smolagents and multiple LLM models. The system creates candidate questions aligned with the Cambridge syllabus, complete with marking schemes and solver algorithms.

## 🎯 Project Goals

- Generate high-quality IGCSE Mathematics practice questions
- Support grade-differentiated difficulty (Grades 1-9)
- Maintain alignment with Cambridge 0580 syllabus
- Include comprehensive marking schemes and solution steps
- Track generation parameters and enable quality review
- Integrate with existing Neon DB and CMS infrastructure

## 🏗️ Architecture

```
├── src/
│   ├── models/           # Pydantic data models
│   ├── database/         # Neon DB client
│   ├── agents/           # smolagents-based generators
│   └── services/         # Orchestration services
├── config/              # Configuration files
├── prompts/             # LLM prompt templates
├── data/                # Syllabus and schema data
└── tests/               # Test scripts
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp env.example .env
# Edit .env with your API keys and database URL
```

### 2. Configure Database

Set your Neon DB connection string in `.env`:
```bash
NEON_DATABASE_URL=postgresql://username:password@host:port/database
```

### 3. Add API Keys

Configure LLM provider keys in `.env`:
```bash
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
ANTHROPIC_API_KEY=your_anthropic_key
```

### 4. Generate Questions

```bash
# Generate basic questions for grades 3, 5, 7
python main.py generate --grades 3 5 7 --count 2

# Generate questions on specific topics
python main.py generate --grades 4 6 --subject-refs C1.6 C1.4 --count 1

# Generate with specific command word
python main.py generate --grades 5 --command-word "Calculate" --count 3

# View generated questions
python main.py list --grade 5 --limit 10

# Show statistics
python main.py stats
```

## 📊 Data Models

### CandidateQuestion
Complete question structure following the schema:
- **Core Data**: Question text, marks, command word
- **Taxonomy**: Topic classification, syllabus references, skill tags
- **Solution**: Marking scheme with criteria and final answers
- **Algorithm**: Step-by-step solution process
- **Metadata**: Generation parameters, model tracking, validation

### Generation Configuration
Tracks all parameters used for generation:
- Target grade and desired marks
- LLM models for each generation step
- Prompt template versions
- Subject content references
- Calculator policy

## 🤖 Agent System

### QuestionGeneratorAgent
- **Input**: Generation configuration, syllabus context
- **Process**: Constructs comprehensive prompts, calls LLM, parses responses
- **Output**: Validated CandidateQuestion objects
- **Features**: Multi-model support, context-aware prompting, validation

### Planned Agents (Phase 2)
- **MarkerAgent**: Specialized marking scheme generation
- **ReviewerAgent**: LLM-based quality review
- **ManagerAgent**: Multi-agent orchestration

## 🗄️ Database Schema

### candidate_questions table
```sql
CREATE TABLE candidate_questions (
    id SERIAL PRIMARY KEY,
    generation_id UUID UNIQUE NOT NULL,
    question_id_global VARCHAR(100) UNIQUE NOT NULL,
    target_grade_input INTEGER NOT NULL,
    marks INTEGER NOT NULL,
    command_word VARCHAR(50) NOT NULL,
    raw_text_content TEXT NOT NULL,

    -- JSON columns for complex data
    taxonomy JSONB NOT NULL,
    solution_and_marking_scheme JSONB NOT NULL,
    solver_algorithm JSONB NOT NULL,

    -- Generation tracking
    llm_model_used_generation VARCHAR(100) NOT NULL,
    prompt_template_version_generation VARCHAR(50) NOT NULL,

    -- Review and status
    status VARCHAR(50) DEFAULT 'candidate',
    reviewer_notes TEXT,
    validation_errors JSONB DEFAULT '[]'::jsonb,

    -- Timestamps
    generation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## 🎚️ Configuration

### Generation Config (`config/generation_config.json`)
```json
{
  "models": {
    "generator": {"default": "gpt-4o"},
    "marker": {"default": "gpt-4o"},
    "reviewer": {"default": "claude-3-5-sonnet"}
  },
  "generation_parameters": {
    "temperature": 0.7,
    "max_tokens": 4000,
    "target_grades": [1,2,3,4,5,6,7,8,9]
  }
}
```

### Prompt Templates (`prompts/`)
- `question_generation_v1.0.txt`: Main generation prompt
- Version controlled for reproducibility
- Includes syllabus context and marking guidelines

## 🧪 Testing

### Basic Functionality Test
```bash
python test_generation.py --mode basic
```

### Detailed Question Analysis
```bash
python test_generation.py --mode detailed
```

### Manual Database Test
```bash
# Run the CLI to generate test questions
python main.py generate --grades 4 --count 1

# Verify in database
python main.py list --grade 4
```

## 📈 Quality Control

### Validation Layers
1. **JSON Schema Validation**: Ensures output structure compliance
2. **Mathematical Consistency**: Checks mark allocation and answer validity
3. **Syllabus Alignment**: Verifies subject content references
4. **Grade Appropriateness**: Validates difficulty estimates

### Review Process
1. **Auto-validation**: Immediate structural and basic content checks
2. **LLM Review**: Automated quality assessment (Phase 2)
3. **Human Review**: Final approval for production use
4. **Status Tracking**: Complete audit trail

## 🔧 Usage Examples

### Generate by Syllabus Topic
```python
from src.services.generation_service import QuestionGenerationService

service = QuestionGenerationService(database_url)
await service.initialize()

response = await service.generate_by_topic(
    subject_content_references=["C1.6", "C1.11"],  # Operations, Ratio
    target_grades=[4, 5, 6],
    count_per_grade=2
)
```

### Batch Generation from Seed
```python
response = await service.generate_batch_from_seed(
    seed_question_id="0580_s15_qp_01_q1a",
    target_grades=[3, 4, 5],
    count_per_grade=3
)
```

### Review Management
```python
success = await service.update_question_review(
    generation_id="uuid-here",
    status="human_reviewed_accepted",
    reviewer_notes="Excellent question, clear and appropriate difficulty"
)
```

## 🚦 Status Codes

- **candidate**: Newly generated, pending review
- **human_reviewed_accepted**: Approved for use
- **human_reviewed_rejected**: Needs improvement
- **llm_reviewed_needs_human**: Flagged by automated review
- **auto_rejected**: Failed basic validation

## 📋 Current Features (Phase 1 - MVP)

✅ **Complete Question Generation Pipeline**
- Single-agent generation with comprehensive prompts
- Multiple LLM model support (GPT-4o, Claude, Gemini)
- Complete schema compliance (taxonomy, marking, algorithm)
- Database integration with full metadata tracking

✅ **CLI Interface**
- Generate questions by grade, topic, or seed
- List and filter generated questions
- Generation statistics and monitoring

✅ **Quality Assurance**
- JSON schema validation
- Mathematical consistency checks
- Comprehensive error reporting
- Version tracking for prompts and models

## 🛣️ Roadmap

### Phase 2: Multi-Agent System
- [ ] Specialized MarkerAgent for marking schemes
- [ ] ReviewerAgent for automated quality checks
- [ ] ManagerAgent for workflow orchestration
- [ ] Inter-agent communication and iteration

### Phase 3: Production Deployment
- [ ] Modal deployment with FastAPI endpoints
- [ ] Batch generation scheduling
- [ ] Payload CMS integration
- [ ] Cloudflare Workers orchestration

### Phase 4: Advanced Features
- [ ] Diagram generation with Manim
- [ ] Adaptive difficulty adjustment
- [ ] Learning analytics integration
- [ ] Multi-language support

## 🔍 Monitoring and Analytics

### Generation Metrics
- Success/failure rates by model and grade
- Average generation time
- Validation error patterns
- Human review acceptance rates

### Quality Metrics
- Grade-appropriate difficulty distribution
- Syllabus coverage completeness
- Mathematical accuracy scores
- Reviewer feedback analysis

## ⚙️ Configuration Management

### Model Selection
Switch between different LLM providers for different tasks:
```python
config = GenerationConfig(
    llm_model_generation=LLMModel.GPT_4O,
    llm_model_marking_scheme=LLMModel.CLAUDE_3_5_SONNET,
    llm_model_review=LLMModel.GEMINI_PRO
)
```

### Prompt Versioning
All prompts are version controlled:
- `prompts/question_generation_v1.0.txt`
- `prompts/marking_scheme_v1.0.txt`
- Database tracks which version was used

### Environment Configuration
Support for different deployment environments:
```bash
# Development
ENVIRONMENT=development
LOG_LEVEL=DEBUG

# Production
ENVIRONMENT=production
LOG_LEVEL=INFO
```

## 🤝 Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure all validation checks pass
5. Test with multiple LLM models when applicable

## 📚 Dependencies

### Core Dependencies
- `smolagents`: Agent framework
- `pydantic`: Data validation and serialization
- `asyncpg`: Async PostgreSQL client
- `openai`: OpenAI API client
- `jsonschema`: JSON validation

### Development Dependencies
- `pytest`: Testing framework
- `black`: Code formatting
- `mypy`: Type checking

## 🐛 Troubleshooting

### Common Issues

**Database Connection Errors**
- Verify `NEON_DATABASE_URL` in `.env`
- Check network connectivity to Neon
- Ensure database credentials are correct

**LLM API Errors**
- Confirm API keys are set correctly
- Check rate limits for your provider
- Verify model names match provider specifications

**Generation Failures**
- Review validation errors in question output
- Check prompt template formatting
- Ensure syllabus data is accessible

**Import Errors**
- Verify all `__init__.py` files are present
- Check Python path includes project root
- Ensure all dependencies are installed

### Debug Mode
```bash
# Run with detailed logging
LOG_LEVEL=DEBUG python main.py generate --grades 4 --count 1

# Test individual components
python test_generation.py --mode detailed
```

This system provides a solid foundation for generating high-quality IGCSE Mathematics questions at scale, with full configurability and quality assurance built in from the start.
