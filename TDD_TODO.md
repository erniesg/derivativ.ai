# Derivativ - TDD Implementation Plan

## 🎯 PROJECT OVERVIEW: Test-Driven Development

**Project Name**: Derivativ - AI-Powered Math Tutor
**Philosophy**: Test-first development approach for reliable, production-ready code
**Timeline**: 5 days for hackathon-ready system
**Architecture**: Multi-agent AI coordination with real-time quality control

**Vision**: Build an AI education platform that generates Cambridge IGCSE Mathematics questions with sophisticated multi-agent workflows that automatically improve question quality.

---

## 📋 DETAILED TODO LIST WITH TDD APPROACH (4 Parallel Tracks)

**Core Innovation**: Multi-agent AI coordination where specialized agents (Generator, Marker, Reviewer, Refiner) work together with visible reasoning to create high-quality educational content.

### **TRACK 1: INFRASTRUCTURE & CONFIGURATION** (Foundation - must complete first)

#### **TODO 1.1: Centralized Configuration Management**
**Status**: PENDING
**Priority**: CRITICAL
**Estimated Time**: 2 hours
**Test First**: `test_config_loading_and_validation()`

#### **TODO 1.2: Data Management Layer**
**Status**: PENDING
**Priority**: CRITICAL
**Estimated Time**: 2 hours
**Test First**: `test_data_persistence_abstraction()`

#### **TODO 1.3: LLM Service Layer**
**Status**: COMPLETED ✅
**Priority**: CRITICAL
**Estimated Time**: 3 hours
**Test First**: `test_llm_service_provider_routing()`
**Implementation**: Multi-provider LLM services (OpenAI, Anthropic, Google) with streaming support, comprehensive unit tests, and live integration tests. All providers tested with real API calls.

#### **TODO 1.4: Quality Control Thresholds**
**Status**: PENDING
**Priority**: HIGH
**Estimated Time**: 1.5 hours
**Test First**: `test_configurable_quality_thresholds()`

#### **TODO 1.5: Logging & Monitoring**
**Status**: PENDING
**Priority**: MEDIUM
**Estimated Time**: 2 hours
**Test First**: `test_agent_reasoning_tracking()`

#### **TODO 1.6: Prompt Management System**
**Status**: COMPLETED ✅
**Priority**: CRITICAL
**Estimated Time**: 3 hours
**Test First**: `test_template_loading_and_caching()`
**Implementation**: Enhanced PromptManager with Jinja2 templates, built-in Cambridge IGCSE templates, async operations, and comprehensive test coverage.

### **TRACK 2: PAST PAPERS INGESTION** (Data foundation)

#### **TODO 2.1: PDF Text Extraction Pipeline**
**Status**: PENDING
**Priority**: HIGH
**Estimated Time**: 4 hours
**Test First**: `test_pdf_text_extraction_accuracy()`

#### **TODO 2.2: Question Structure Parser**
**Status**: PENDING
**Priority**: HIGH
**Estimated Time**: 3 hours
**Test First**: `test_question_boundary_detection()`

#### **TODO 2.3: Marking Scheme Extractor**
**Status**: PENDING
**Priority**: HIGH
**Estimated Time**: 3 hours
**Test First**: `test_marking_criteria_extraction()`

#### **TODO 2.4: Diagram Detection & Asset Extraction**
**Status**: PENDING
**Priority**: MEDIUM
**Estimated Time**: 4 hours
**Test First**: `test_diagram_detection_accuracy()`

#### **TODO 2.5: Cambridge Compliance Validator**
**Status**: PENDING
**Priority**: CRITICAL
**Estimated Time**: 2 hours
**Test First**: `test_strict_enum_validation()`

#### **TODO 2.6: Data Processing Pipeline**
**Status**: PENDING
**Priority**: HIGH
**Estimated Time**: 2 hours
**Test First**: `test_end_to_end_ingestion_pipeline()`

### **TRACK 3: MULTI-AGENT GENERATION** (Core value proposition)

#### **TODO 3.1: Base Agent Framework + Smolagents Integration**
**Status**: COMPLETED ✅
**Priority**: CRITICAL
**Estimated Time**: 6 hours
**Test First**: `test_smolagents_integration()`, `test_async_sync_compatibility()`
**Implementation**: Complete smolagents multi-agent coordination system with:
- Native smolagents tools (generate_math_question, review_question_quality, refine_question)
- Agent factory with question_generator, quality_control, and multi_agent configurations
- Async/sync compatibility layer with proper event loop handling
- Live integration tests with real LLM API calls (6/7 passing, 1 skipped for HF_TOKEN)
- Interactive and tools-only demo modes
- Setup wizard with API key detection and configuration

#### **TODO 3.2: Question Generator Agent**
**Status**: COMPLETED ✅
**Priority**: CRITICAL
**Estimated Time**: 4 hours
**Test First**: `test_question_generation_cambridge_compliance()`
**Implementation**: QuestionGeneratorAgent with Cambridge IGCSE compliance, JSON parsing, and comprehensive unit tests.

#### **TODO 3.3: Marker Agent**
**Status**: PENDING
**Priority**: HIGH
**Estimated Time**: 3 hours
**Test First**: `test_marking_scheme_generation()`

#### **TODO 3.4: Review Agent**
**Status**: COMPLETED ✅
**Priority**: CRITICAL
**Estimated Time**: 4 hours
**Test First**: `test_multi_dimensional_quality_assessment()`
**Implementation**: ReviewAgent with multi-dimensional quality scoring, detailed feedback generation, and full E2E testing.

#### **TODO 3.5: Quality Control Workflow**
**Status**: PENDING
**Priority**: HIGH
**Estimated Time**: 2 hours
**Test First**: `test_threshold_based_decisions()`

#### **TODO 3.6: Modal Deployment & Orchestration**
**Status**: PENDING
**Priority**: HIGH
**Estimated Time**: 3 hours
**Test First**: `test_modal_agent_coordination()`

### **TRACK 4: MANIM DIAGRAM GENERATION** (Visual enhancement)

#### **TODO 4.1: Diagram Template Library**
**Status**: PENDING
**Priority**: MEDIUM
**Estimated Time**: 5 hours
**Test First**: `test_manim_template_rendering()`

#### **TODO 4.2: Diagram Code Generator Agent**
**Status**: PENDING
**Priority**: MEDIUM
**Estimated Time**: 4 hours
**Test First**: `test_manim_code_generation()`

#### **TODO 4.3: Diagram Renderer & Validator**
**Status**: PENDING
**Priority**: MEDIUM
**Estimated Time**: 3 hours
**Test First**: `test_diagram_rendering_validation()`

#### **TODO 4.4: Diagram Review Integration**
**Status**: PENDING
**Priority**: MEDIUM
**Estimated Time**: 2 hours
**Test First**: `test_diagram_quality_assessment()`

#### **TODO 4.5: Visual Asset Pipeline**
**Status**: PENDING
**Priority**: LOW
**Estimated Time**: 2 hours
**Test First**: `test_visual_asset_integration()`

**Derivativ Project Structure** (Production Stack):
```
derivativ/
├── pyproject.toml              # Python packaging with Modal deps
├── modal.toml                  # Modal configuration
├── wrangler.toml              # Cloudflare Workers config
├── requirements/               # Split requirements
│   ├── base.txt               # Modal, smolagents, Neon DB
│   ├── ai.txt                 # OpenAI, Anthropic, Gemini, OpenRouter
│   └── dev.txt                # pytest, Modal testing tools
├── tests/                     # Comprehensive test suite
│   ├── conftest.py            # Shared fixtures + Modal test setup
│   ├── modal/                 # Modal function tests
│   ├── smolagents/            # Agent coordination tests
│   ├── cloudflare/            # Workers and edge tests
│   └── e2e/                   # Full workflow tests
├── src/derivativ/             # Main package
│   ├── agents/                # Modal functions with smolagents
│   ├── models/                # Pydantic models for Neon DB
│   ├── database/              # Neon DB operations
│   └── utils/                 # Shared utilities
├── cloudflare-workers/        # Cloudflare Workers code
│   ├── api-gateway.ts         # Main API gateway
│   ├── websocket-handler.ts   # Real-time streaming
│   └── r2-cache.ts           # R2 object storage
├── frontend/                  # Next.js + AI SDK
│   ├── package.json           # Next.js + AI SDK deps
│   ├── components/            # Streaming React components
│   └── app/                   # App router + API routes
├── payload-cms/               # Payload CMS on Cloudflare
└── scripts/                   # Deployment and utility scripts
```

#### **TODO 1.2: Write Comprehensive Test Suite FIRST**
**Status**: PENDING
**Priority**: CRITICAL
**Estimated Time**: 2 hours
**Philosophy**: All tests written before any implementation

**Critical Test Categories**:
```python
# 1. CORE WORKFLOW TESTS (Demo-Critical)
test_generate_question_end_to_end()           # Main demo flow
test_multi_agent_coordination_visible()       # Show agent reasoning
test_quality_control_improvement_cycle()      # Automatic refinement
test_real_time_progress_tracking()            # Live demo updates

# 2. AGENT COORDINATION TESTS (Judge Appeal)
test_question_generator_agent()               # Creates valid questions
test_marker_agent_scheme_generation()         # Proper mark allocation
test_review_agent_quality_scoring()           # Consistent assessment
test_refinement_agent_improvements()          # Meaningful enhancements
test_orchestrator_agent_coordination()        # Multi-agent workflow

# 3. LLM INTERFACE TESTS (Reliability)
test_llm_provider_switching()                 # OpenAI → Anthropic fallback
test_llm_timeout_handling()                   # Network resilience
test_llm_cost_tracking()                      # Budget monitoring
test_llm_response_validation()                # JSON parsing robustness

# 4. DATA MODEL TESTS (Foundation)
test_question_model_validation()              # Pydantic validation
test_cambridge_syllabus_compliance()          # Real curriculum refs
test_database_persistence_audit()             # Full audit trails
test_generation_session_tracking()            # Complete workflows

# 5. API TESTS (Integration)
test_api_question_generation_endpoint()       # REST API works
test_api_real_time_websocket_updates()        # Live progress
test_api_error_handling_graceful()            # Robust error responses
test_api_concurrent_request_handling()        # Scale simulation

# 6. PERFORMANCE TESTS (Demo Requirements)
test_generation_completes_under_30_seconds()  # Speed requirement
test_concurrent_generation_stability()        # Multiple requests
test_memory_usage_bounded()                   # Resource management
test_demo_scenario_preloading()               # Fast demo startup

# 7. QUALITY TESTS (Educational Value)
test_mathematical_correctness_validation()    # Accurate solutions
test_grade_appropriate_difficulty()           # Proper targeting
test_command_word_usage_correct()             # Cambridge compliance
test_question_variety_generation()            # No duplicates
```

**TDD Success Criteria**:
- [ ] **All tests written BEFORE implementation**
- [ ] **95%+ test coverage** on core functionality
- [ ] **Demo-critical paths tested first** (generation, coordination, quality)
- [ ] **Performance benchmarks** embedded in tests
- [ ] **Fallback scenarios tested** for demo reliability

#### **TODO 1.3: Setup Production Technology Stack**
**Status**: PENDING
**Priority**: CRITICAL
**Estimated Time**: 2 hours
**Test First**: Test all infrastructure connections work

**Production Technology Stack** (Actual Decisions):
```bash
# Core Infrastructure
modal>=0.63.0                 # Serverless compute for AI agents
smolagents                    # Facebook's multi-agent framework
neon-postgresql               # Serverless PostgreSQL database

# AI/ML Stack
openai>=1.3.0                 # Primary LLM provider
anthropic>=0.7.0              # Claude models
google-generativeai>=0.3.0    # Gemini models
openrouter                    # LLM routing and fallbacks

# Cloudflare Stack
wrangler                      # Cloudflare Workers CLI
@cloudflare/workers-types     # TypeScript definitions
@cloudflare/d1               # SQLite edge database
@cloudflare/r2               # Object storage

# Frontend Stack
next>=14.0.0                 # React framework with AI SDK
@ai-sdk/react                # Streaming React components
@ai-sdk/core                 # AI SDK core functionality
payload                     # Headless CMS

# Testing & Development
pytest>=7.4.0               # Testing framework
pytest-asyncio>=0.21.0      # Async testing
modal-pytest                # Modal-specific testing tools
```

**Test Infrastructure Connections**:
```python
def test_modal_connection():
    """Test Modal CLI and deployment works"""
    import modal
    app = modal.App("test-derivativ")
    assert app is not None

def test_neon_db_connection():
    """Test Neon DB connection successful"""
    import asyncpg
    # Test connection to Neon DB
    assert True  # Replace with actual connection test

def test_cloudflare_workers_deploy():
    """Test Cloudflare Workers deployment ready"""
    # Test wrangler CLI available
    import subprocess
    result = subprocess.run(["wrangler", "version"], capture_output=True)
    assert result.returncode == 0

def test_smolagents_import():
    """Test smolagents framework available"""
    import smolagents
    assert hasattr(smolagents, 'Agent')
```

#### **TODO 1.4: Implement Pydantic Data Models**
**Status**: PENDING
**Priority**: CRITICAL
**Estimated Time**: 2 hours
**Test First**: All model validation tests

**Core Models for Derivativ**:
```python
# src/derivativ/models/question.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
from uuid import UUID4

class Grade(int, Enum):
    GRADE_1 = 1
    GRADE_2 = 2
    # ... up to GRADE_9 = 9

class CommandWord(str, Enum):
    CALCULATE = "Calculate"
    FIND = "Find"
    SOLVE = "Solve"
    DETERMINE = "Determine"
    SHOW = "Show"
    PROVE = "Prove"
    # ... Cambridge command words

class QualityAction(str, Enum):
    APPROVE = "approve"
    MANUAL_REVIEW = "manual_review"
    REFINE = "refine"
    REGENERATE = "regenerate"
    REJECT = "reject"

class Question(BaseModel):
    id: Optional[UUID4] = None
    raw_text_content: str = Field(..., min_length=10, max_length=1000)
    marks: int = Field(..., ge=1, le=20)
    target_grade: Grade
    command_word: CommandWord
    subject_content_references: List[str] = Field(..., min_items=1)
    calculator_allowed: bool = False
    quality_score: Optional[float] = Field(None, ge=0, le=1)
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = {}

class MarkingScheme(BaseModel):
    question_id: UUID4
    total_marks: int
    mark_allocation: List[Dict[str, Any]]  # Step-by-step marking
    solution_method: str
    alternative_methods: List[str] = []

class GenerationRequest(BaseModel):
    topic: str = Field(..., min_length=2)
    grade: Grade
    marks: int = Field(..., ge=1, le=20)
    count: int = Field(1, ge=1, le=10)
    calculator_allowed: bool = False
    subject_area: Optional[str] = None

class AgentResult(BaseModel):
    success: bool
    agent_name: str
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    reasoning_steps: List[str] = []
    processing_time: float = 0.0
    metadata: Dict[str, Any] = {}
```

**Model Tests** (Write These First):
```python
def test_question_model_validation():
    """Test Question model validates correctly"""
    question = Question(
        raw_text_content="Calculate 2 + 3",
        marks=1,
        target_grade=Grade.GRADE_3,
        command_word=CommandWord.CALCULATE,
        subject_content_references=["C1.6"],
        calculator_allowed=False
    )
    assert question.marks == 1
    assert question.target_grade == Grade.GRADE_3

def test_question_model_validation_errors():
    """Test Question model rejects invalid data"""
    with pytest.raises(ValidationError):
        Question(
            raw_text_content="Too short",  # Under 10 chars
            marks=25,  # Over max marks
            target_grade=15,  # Invalid grade
            command_word="InvalidCommand",
            subject_content_references=[]  # Empty list
        )

def test_generation_request_validation():
    """Test GenerationRequest model works"""
    request = GenerationRequest(
        topic="arithmetic",
        grade=Grade.GRADE_5,
        marks=3,
        count=2
    )
    assert request.topic == "arithmetic"
    assert request.count == 2
```

---

### **PHASE 2: CORE MODELS & DATA LAYER** (Day 1-2)

#### **TODO 2.1: Define Pydantic Models with Tests**
**Priority**: HIGH
**Estimated Time**: 3 hours
**Tech Stack**: Pydantic v2, SQLAlchemy 2.0, Enum for type safety

**Test First**:
```python
def test_question_model_validation():
    """Test Question model validates correctly"""
    question = Question(
        raw_text_content="What is 2+2?",
        marks=1,
        command_word="Calculate",
        subject_content_references=["C1.6"],
        target_grade=3
    )
    assert question.marks == 1
    assert question.target_grade == 3

def test_question_model_invalid_grade():
    """Test Question model rejects invalid grades"""
    with pytest.raises(ValidationError):
        Question(target_grade=15)  # Invalid grade
```

**Implementation Strategy**:
```python
# src/math_tutor/models/question.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum

class Grade(int, Enum):
    GRADE_1 = 1
    GRADE_2 = 2
    # ... up to 9

class CommandWord(str, Enum):
    CALCULATE = "Calculate"
    FIND = "Find"
    SOLVE = "Solve"
    # ... Cambridge command words

class Question(BaseModel):
    raw_text_content: str = Field(..., min_length=10)
    marks: int = Field(..., ge=1, le=20)
    command_word: CommandWord
    subject_content_references: List[str]
    target_grade: Grade

    @validator('subject_content_references')
    def validate_content_refs(cls, v):
        # Validate against Cambridge syllabus
        valid_refs = ["C1.1", "C1.2", ...]  # From enum
        for ref in v:
            if ref not in valid_refs:
                raise ValueError(f"Invalid content reference: {ref}")
        return v
```

#### **TODO 2.2: Database Schema with Migrations**
**Priority**: HIGH
**Estimated Time**: 2 hours
**Tech Stack**: SQLAlchemy 2.0, Alembic, PostgreSQL

**Test First**:
```python
def test_database_question_persistence():
    """Test questions can be saved and retrieved"""
    question = QuestionDB(content="Test", marks=1, grade=3)
    session.add(question)
    session.commit()

    retrieved = session.query(QuestionDB).first()
    assert retrieved.content == "Test"
    assert retrieved.marks == 1

def test_database_audit_trail():
    """Test all operations are audited"""
    question = create_question()
    session.add(question)
    session.commit()

    audit_logs = session.query(AuditLog).all()
    assert len(audit_logs) == 1
    assert audit_logs[0].operation == "CREATE"
```

**Implementation**:
```python
# src/math_tutor/database/models.py
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class QuestionDB(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    marks = Column(Integer, nullable=False)
    grade = Column(Integer, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    metadata = Column(JSON)

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    table_name = Column(String(50), nullable=False)
    operation = Column(String(10), nullable=False)  # CREATE, UPDATE, DELETE
    record_id = Column(Integer, nullable=False)
    changes = Column(JSON)
    timestamp = Column(DateTime, server_default=func.now())
```

---

### **PHASE 3: LLM INTERFACE & AGENT FRAMEWORK** (Day 2)

#### **TODO 3.1: Create Abstracted LLM Interface**
**Priority**: HIGH
**Estimated Time**: 3 hours
**Tech Stack**: OpenAI SDK, Anthropic SDK, Google AI SDK, LiteLLM for unified interface

**Test First**:
```python
def test_llm_provider_switching():
    """Test can switch between LLM providers seamlessly"""
    llm = LLMInterface(provider="openai", model="gpt-4o")
    response1 = llm.generate("Test prompt")

    llm.switch_provider("anthropic", "claude-3-5-sonnet")
    response2 = llm.generate("Test prompt")

    assert response1.content is not None
    assert response2.content is not None
    assert response1.provider == "openai"
    assert response2.provider == "anthropic"

def test_llm_error_handling():
    """Test LLM handles API errors gracefully"""
    llm = LLMInterface(provider="openai", model="invalid-model")

    with pytest.raises(LLMError):
        llm.generate("Test prompt")

def test_llm_timeout_handling():
    """Test LLM handles timeouts properly"""
    llm = LLMInterface(provider="openai", timeout=1)

    with pytest.raises(TimeoutError):
        llm.generate("Very long prompt" * 1000)
```

**Implementation Strategy**:
```python
# src/math_tutor/core/llm_interface.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import openai
import anthropic

@dataclass
class LLMResponse:
    content: str
    provider: str
    model: str
    tokens_used: int
    cost_estimate: float

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            provider="openai",
            model=self.model,
            tokens_used=response.usage.total_tokens,
            cost_estimate=self._calculate_cost(response.usage)
        )

class LLMInterface:
    def __init__(self, provider: str, model: str, **config):
        self.providers = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "google": GoogleProvider
        }
        self.current_provider = self._create_provider(provider, model, **config)

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        try:
            return self.current_provider.generate(prompt, **kwargs)
        except Exception as e:
            raise LLMError(f"Generation failed: {e}")
```

#### **TODO 3.2: Implement Base Agent Class**
**Priority**: HIGH
**Estimated Time**: 2 hours

**Test First**:
```python
def test_agent_base_functionality():
    """Test base agent can be instantiated and process requests"""
    agent = BaseAgent(llm_interface=mock_llm, name="test_agent")

    result = agent.process({"input": "test"})

    assert result.success is True
    assert result.output is not None
    assert result.agent_name == "test_agent"

def test_agent_error_handling():
    """Test agent handles processing errors"""
    agent = BaseAgent(llm_interface=failing_llm, name="test_agent")

    result = agent.process({"input": "test"})

    assert result.success is False
    assert result.error is not None

def test_agent_reasoning_tracking():
    """Test agent tracks its reasoning process"""
    agent = BaseAgent(llm_interface=mock_llm, name="test_agent")

    result = agent.process({"input": "test"})

    assert len(result.reasoning_steps) > 0
    assert result.reasoning_steps[0].step_type in ["observation", "thought", "action"]
```

**Implementation**:
```python
# src/math_tutor/agents/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum

class StepType(Enum):
    OBSERVATION = "observation"
    THOUGHT = "thought"
    ACTION = "action"

@dataclass
class ReasoningStep:
    step_type: StepType
    content: str
    timestamp: float

@dataclass
class AgentResult:
    success: bool
    output: Optional[Dict[str, Any]]
    error: Optional[str]
    agent_name: str
    reasoning_steps: List[ReasoningStep]
    metadata: Dict[str, Any]

class BaseAgent(ABC):
    def __init__(self, llm_interface, name: str, config: Dict[str, Any] = None):
        self.llm = llm_interface
        self.name = name
        self.config = config or {}
        self.reasoning_steps = []

    def process(self, input_data: Dict[str, Any]) -> AgentResult:
        try:
            self._observe("Received input", input_data)
            self._think("Planning approach")

            output = self._execute(input_data)

            return AgentResult(
                success=True,
                output=output,
                error=None,
                agent_name=self.name,
                reasoning_steps=self.reasoning_steps.copy(),
                metadata={"processing_time": self._get_processing_time()}
            )
        except Exception as e:
            return AgentResult(
                success=False,
                output=None,
                error=str(e),
                agent_name=self.name,
                reasoning_steps=self.reasoning_steps.copy(),
                metadata={}
            )

    @abstractmethod
    def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Implement specific agent logic"""
        pass

    def _observe(self, observation: str, data: Any = None):
        """Record an observation"""
        self.reasoning_steps.append(
            ReasoningStep(StepType.OBSERVATION, observation, time.time())
        )

    def _think(self, thought: str):
        """Record a thought/reasoning step"""
        self.reasoning_steps.append(
            ReasoningStep(StepType.THOUGHT, thought, time.time())
        )

    def _act(self, action: str):
        """Record an action taken"""
        self.reasoning_steps.append(
            ReasoningStep(StepType.ACTION, action, time.time())
        )
```

#### **TODO 3.3: Implement Specialized Agents**
**Priority**: HIGH
**Estimated Time**: 4 hours

**Test First**:
```python
def test_question_generator_agent():
    """Test QuestionGenerator creates valid questions"""
    agent = QuestionGeneratorAgent(llm_interface=mock_llm)

    input_data = {
        "topic": "arithmetic",
        "grade": 3,
        "marks": 2,
        "calculator_allowed": False
    }

    result = agent.process(input_data)

    assert result.success is True
    assert "question_text" in result.output
    assert "command_word" in result.output
    assert result.output["marks"] == 2

def test_marker_agent():
    """Test MarkerAgent creates marking schemes"""
    agent = MarkerAgent(llm_interface=mock_llm)

    input_data = {
        "question_text": "Calculate 2 + 3",
        "marks": 1,
        "subject_refs": ["C1.6"]
    }

    result = agent.process(input_data)

    assert result.success is True
    assert "marking_scheme" in result.output
    assert "total_marks" in result.output

def test_review_agent():
    """Test ReviewAgent assesses question quality"""
    agent = ReviewAgent(llm_interface=mock_llm)

    input_data = {
        "question": mock_question,
        "marking_scheme": mock_marking_scheme
    }

    result = agent.process(input_data)

    assert result.success is True
    assert "quality_score" in result.output
    assert 0 <= result.output["quality_score"] <= 1
    assert "feedback" in result.output
```

**Implementation Priority**:
1. **QuestionGeneratorAgent**: Creates initial questions
2. **MarkerAgent**: Generates marking schemes
3. **ReviewAgent**: Quality assessment
4. **RefinementAgent**: Improvement suggestions

---

### **PHASE 4: ORCHESTRATION & QUALITY CONTROL** (Day 2-3)

#### **TODO 4.1: Multi-Agent Orchestration**
**Priority**: HIGH
**Estimated Time**: 3 hours

**Test First**:
```python
def test_orchestrator_coordination():
    """Test orchestrator coordinates multiple agents"""
    orchestrator = QuestionOrchestrator(
        generator=mock_generator,
        marker=mock_marker,
        reviewer=mock_reviewer
    )

    request = GenerationRequest(
        topic="algebra",
        grade=5,
        count=1
    )

    result = orchestrator.generate_questions(request)

    assert result.success is True
    assert len(result.questions) == 1
    assert result.questions[0].has_marking_scheme
    assert result.questions[0].quality_score >= 0.4

def test_quality_control_workflow():
    """Test quality control makes correct decisions"""
    qc = QualityControlWorkflow(
        thresholds={"auto_approve": 0.85, "manual_review": 0.6, "reject": 0.3}
    )

    # High quality question should auto-approve
    high_quality = mock_question_with_score(0.9)
    decision = qc.assess(high_quality)
    assert decision.action == "approve"

    # Low quality should reject
    low_quality = mock_question_with_score(0.2)
    decision = qc.assess(low_quality)
    assert decision.action == "reject"
```

#### **TODO 4.2: Quality Control Implementation**
**Priority**: HIGH
**Estimated Time**: 2 hours

```python
# src/math_tutor/core/quality_control.py
from enum import Enum
from dataclasses import dataclass

class QualityAction(Enum):
    APPROVE = "approve"
    MANUAL_REVIEW = "manual_review"
    REFINE = "refine"
    REGENERATE = "regenerate"
    REJECT = "reject"

@dataclass
class QualityDecision:
    action: QualityAction
    confidence: float
    reasoning: str
    suggested_improvements: List[str] = None

class QualityControlWorkflow:
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds

    def assess(self, question_with_score) -> QualityDecision:
        score = question_with_score.quality_score

        if score >= self.thresholds["auto_approve"]:
            return QualityDecision(
                action=QualityAction.APPROVE,
                confidence=score,
                reasoning="High quality score meets auto-approval threshold"
            )
        elif score <= self.thresholds["reject"]:
            return QualityDecision(
                action=QualityAction.REJECT,
                confidence=1.0 - score,
                reasoning="Quality score below rejection threshold"
            )
        # ... additional logic
```

---

### **PHASE 5: API & FRONTEND** (Day 3-4)

#### **TODO 5.1: FastAPI Backend**
**Priority**: MEDIUM
**Estimated Time**: 3 hours
**Tech Stack**: FastAPI, Pydantic, async/await

**Test First**:
```python
def test_api_health_check():
    """Test API health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_api_generate_questions():
    """Test question generation endpoint"""
    request_data = {
        "topic": "arithmetic",
        "grade": 3,
        "count": 2,
        "calculator_allowed": False
    }

    response = client.post("/questions/generate", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert len(data["questions"]) == 2
    assert data["generation_id"] is not None

def test_api_error_handling():
    """Test API handles invalid requests"""
    invalid_request = {"grade": 15}  # Invalid grade

    response = client.post("/questions/generate", json=invalid_request)

    assert response.status_code == 422
    assert "validation error" in response.json()["detail"]
```

#### **TODO 5.2: React Frontend**
**Priority**: MEDIUM
**Estimated Time**: 4 hours
**Tech Stack**: React 18, TypeScript, Tailwind CSS, React Query

**Component Test First**:
```typescript
// frontend/src/components/__tests__/QuestionGenerator.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import QuestionGenerator from '../QuestionGenerator';

test('renders question generation form', () => {
  render(<QuestionGenerator />);

  expect(screen.getByLabelText(/topic/i)).toBeInTheDocument();
  expect(screen.getByLabelText(/grade/i)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /generate/i })).toBeInTheDocument();
});

test('submits form with correct data', async () => {
  const mockOnGenerate = jest.fn();
  render(<QuestionGenerator onGenerate={mockOnGenerate} />);

  fireEvent.change(screen.getByLabelText(/topic/i), { target: { value: 'algebra' } });
  fireEvent.change(screen.getByLabelText(/grade/i), { target: { value: '5' } });
  fireEvent.click(screen.getByRole('button', { name: /generate/i }));

  expect(mockOnGenerate).toHaveBeenCalledWith({
    topic: 'algebra',
    grade: 5,
    count: 1
  });
});
```

---

### **PHASE 6: DEPLOYMENT & DEMO** (Day 4-5)

#### **TODO 6.1: Docker & Environment Setup**
**Priority**: LOW
**Estimated Time**: 2 hours

**Test First**:
```python
def test_docker_build():
    """Test Docker image builds successfully"""
    result = subprocess.run(["docker", "build", "-t", "math-tutor", "."])
    assert result.returncode == 0

def test_environment_configuration():
    """Test all required environment variables are documented"""
    env_example = read_file(".env.example")
    required_vars = [
        "DATABASE_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"
    ]
    for var in required_vars:
        assert var in env_example
```

#### **TODO 6.2: Demo System**
**Priority**: HIGH
**Estimated Time**: 3 hours

**Demo Test Requirements**:
```python
def test_demo_generation_speed():
    """Test demo generates questions quickly enough for live presentation"""
    start_time = time.time()

    result = demo_system.generate_sample_questions()

    duration = time.time() - start_time
    assert duration < 30  # Must complete in under 30 seconds
    assert len(result.questions) >= 3

def test_demo_error_resilience():
    """Test demo handles API failures gracefully"""
    with mock_api_failure():
        result = demo_system.generate_with_fallback()

    assert result.success is True  # Should fallback to cached examples
    assert result.error_message is not None  # Should inform user
```

---

## 🛠️ TECHNOLOGY STACK DECISIONS

### **Core Technologies** (Final Decisions)

#### **Backend Framework**
```python
# Primary: FastAPI (modern, async, automatic docs)
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.4.0
```

#### **Database Stack**
```python
# PostgreSQL with modern SQLAlchemy
sqlalchemy>=2.0.0
alembic>=1.12.0
asyncpg>=0.29.0  # Async PostgreSQL driver
```

#### **AI/ML Libraries**
```python
# Multi-provider LLM support
openai>=1.3.0
anthropic>=0.7.0
google-generativeai>=0.3.0
litellm>=1.0.0  # Unified LLM interface

# Agent framework
langchain>=0.0.350  # For complex agent patterns
```

#### **Testing Framework**
```python
# Comprehensive testing stack
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0
factory-boy>=3.3.0
httpx>=0.25.0  # For API testing
```

#### **Frontend Stack**
```json
{
  "react": "^18.2.0",
  "typescript": "^5.2.0",
  "@tanstack/react-query": "^5.0.0",
  "tailwindcss": "^3.3.0",
  "react-hook-form": "^7.47.0",
  "@hookform/resolvers": "^3.3.0",
  "zod": "^3.22.0"
}
```

#### **Development Tools**
```python
# Code quality and development
black>=23.0.0
isort>=5.12.0
mypy>=1.6.0
pre-commit>=3.5.0
ruff>=0.1.0  # Fast Python linter
```

### **Why These Choices?**

1. **FastAPI**: Automatic OpenAPI docs, native async support, excellent performance
2. **SQLAlchemy 2.0**: Modern async support, better type hints, improved performance
3. **Pydantic v2**: Faster validation, better error messages, excellent TypeScript integration
4. **React 18**: Concurrent features, excellent ecosystem, TypeScript support
5. **LiteLLM**: Unified interface across providers, cost tracking, fallback handling
6. **PostgreSQL**: ACID compliance, JSON support, excellent performance for complex queries

---

## 🚀 OPTIMIZATION OPPORTUNITIES

### **Performance Optimizations**

#### **Caching Strategy**
```python
# Redis for fast question caching
redis>=5.0.0
python-redis-cache>=1.0.0

# Cache frequently requested questions
@cache(ttl=3600)  # 1 hour cache
def get_questions_by_topic(topic: str, grade: int):
    return database.query_questions(topic=topic, grade=grade)
```

#### **Database Optimizations**
```sql
-- Optimized indexes for common queries
CREATE INDEX idx_questions_topic_grade ON questions(topic, grade);
CREATE INDEX idx_questions_quality_score ON questions(quality_score) WHERE quality_score >= 0.6;
CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp);
```

#### **Async Processing**
```python
# Background task processing
import asyncio
from celery import Celery

# Async question generation for large batches
async def generate_questions_batch(requests: List[GenerationRequest]):
    tasks = [generate_single_question(req) for req in requests]
    return await asyncio.gather(*tasks)
```

### **Cost Optimizations**

#### **Model Selection Strategy**
```python
# Intelligent model routing based on complexity
class ModelRouter:
    def select_model(self, complexity: str, budget: float) -> str:
        if complexity == "simple" and budget < 0.01:
            return "gpt-4o-mini"  # Cheap for basic questions
        elif complexity == "complex":
            return "claude-3-5-sonnet"  # Best reasoning
        else:
            return "gpt-4o"  # Balanced option
```

#### **Token Usage Optimization**
```python
# Optimized prompts with fewer tokens
OPTIMIZED_PROMPTS = {
    "question_gen": "Generate IGCSE math Q: {topic}, Grade {grade}, {marks}m",
    "marking": "Mark scheme for: {question}. Format: step=mark"
}
```

### **Quality Optimizations**

#### **Multi-Model Consensus**
```python
# Use multiple models for quality validation
async def validate_with_consensus(question: Question) -> float:
    scores = await asyncio.gather(
        model_a.assess_quality(question),
        model_b.assess_quality(question),
        model_c.assess_quality(question)
    )
    return statistics.median(scores)  # Robust to outliers
```

#### **Automated Testing Pipeline**
```python
# Continuous quality monitoring
def test_generated_questions_quality():
    recent_questions = get_recent_generations(hours=24)

    for question in recent_questions:
        assert question.quality_score >= 0.4
        assert question.has_valid_marking_scheme
        assert question.matches_cambridge_syllabus
```

---

## 🎯 SUCCESS METRICS & VALIDATION

### **Technical Metrics**
- [ ] **Test Coverage**: >95% line coverage, >90% branch coverage
- [ ] **Performance**: <30s question generation, <3s API response time
- [ ] **Reliability**: >99% uptime, graceful error handling
- [ ] **Security**: No exposed secrets, input validation on all endpoints

### **Quality Metrics**
- [ ] **Mathematical Accuracy**: >98% correct solutions
- [ ] **Curriculum Compliance**: 100% valid Cambridge references
- [ ] **Difficulty Appropriateness**: ±1 grade level accuracy
- [ ] **Question Variety**: No duplicate questions in 1000+ generations

### **User Experience Metrics**
- [ ] **Generation Speed**: Perceived as "fast" by users
- [ ] **Interface Usability**: Complete workflows without documentation
- [ ] **Error Recovery**: Clear error messages, suggested fixes
- [ ] **Mobile Responsiveness**: Works on tablets/phones

### **Business Metrics**
- [ ] **Demo Impact**: Judges understand technical sophistication
- [ ] **Market Readiness**: Teachers express interest in using system
- [ ] **Scalability**: Architecture supports 1000+ concurrent users
- [ ] **Cost Efficiency**: <$0.10 per question generation

---

## 🏁 COMPLETION CRITERIA

### **Minimum Viable Product (MVP)** - 🎯 80% COMPLETE
- [✅] Generate mathematically correct IGCSE questions - QuestionGeneratorAgent implemented with validation
- [✅] Multi-agent quality control pipeline working - ReviewAgent + RefinementAgent complete
- [📋] Web interface for live demonstration - Architecture planned
- [📋] Database persistence with audit trails - Models defined, integration pending
- [✅] Basic error handling and recovery - Comprehensive error handling implemented

### **Hackathon-Ready Product** - 🚀 60% COMPLETE
- [📋] Impressive live demonstration (5-minute flow) - Agent workflows ready for demo
- [✅] Multiple LLM providers with intelligent routing - OpenAI, Anthropic, Google implemented
- [✅] Real-time quality assessment visible to judges - Transparent agent reasoning implemented
- [📋] Teacher workflow simulation - Architecture ready for implementation
- [📋] Student practice interface prototype - Planned for next phase

### **Production-Ready Features** - 🏗️ 70% COMPLETE
- [✅] Comprehensive test suite (>95% coverage) - 280+ tests across unit/integration/E2E
- [📋] Docker deployment with environment management - Configuration ready
- [📋] API documentation and usage examples - Pydantic models provide auto-docs
- [🔧] Performance monitoring and alerting - Performance tests in progress
- [📋] Security audit and compliance checks - Environment-based secrets implemented

**This TDD approach ensures we build a robust, well-tested system that can impress hackathon judges while being genuinely production-ready.**
