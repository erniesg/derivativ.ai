# Test Organization

## 🏗️ Test Structure

Our tests are organized by **test type** to separate fast feedback from comprehensive testing:

```
tests/
├── unit/              # ⚡ Fast, isolated tests
├── integration/       # 🔗 Service interaction tests
├── e2e/              # 🌐 End-to-end workflow tests
└── performance/      # 🚀 Load/stress tests
```

## 🎯 Test Types & Usage

### ⚡ Unit Tests (Mission Critical)
**Purpose**: Test individual functions/methods in isolation
**Speed**: <1 second each
**When to run**: Every commit, CI/CD pipeline

```bash
# Run only unit tests (fastest feedback)
pytest tests/unit/ -v

# Run specific unit test file
pytest tests/unit/test_question_generator_unit.py -v
```

**What they test**:
- Input/output contracts
- Type validation
- Error handling
- Core business logic

### 🔗 Integration Tests (Important)
**Purpose**: Test services working together
**Speed**: 1-5 seconds each
**When to run**: Before merging, nightly builds

```bash
# Run integration tests
pytest tests/integration/ -v

# Run with services
pytest tests/integration/test_agent_integration.py -v
```

**What they test**:
- Agent + LLM service interaction
- Prompt manager template rendering
- JSON parser extraction
- Service configuration

### 🌐 E2E Tests (Important)
**Purpose**: Test complete user workflows
**Speed**: 5-30 seconds each
**When to run**: Before releases, weekly

```bash
# Run end-to-end tests
pytest tests/e2e/ -v

# Test complete workflows
pytest tests/e2e/test_agent_workflow.py -v
```

**What they test**:
- Question generation → Marking scheme workflow
- Multi-agent coordination
- Different tier/grade scenarios
- Real user scenarios

### 🚀 Performance Tests (Nice-to-have)
**Purpose**: Test load, stress, timing
**Speed**: 30+ seconds each
**When to run**: Before major releases, performance tuning

```bash
# Run performance tests (slowest)
pytest tests/performance/ -v

# Test with timeout
pytest tests/performance/ -v --timeout=60
```

**What they test**:
- Concurrent request handling
- Timeout scenarios
- Retry logic performance
- Large batch processing

## 🚨 Test Criticality

### 🔴 Must Pass (Deployment Blockers)
```bash
# Mission-critical tests only
pytest tests/unit/ -v
```

### 🟡 Should Pass (Pre-release)
```bash
# Important tests
pytest tests/unit/ tests/integration/ tests/e2e/ -v
```

### 🟢 Can Fail (Performance tuning)
```bash
# All tests including performance
pytest tests/ -v
```

## 📊 Quick Commands

```bash
# Fastest feedback (30 seconds)
pytest tests/unit/ -v

# Pre-commit checks (2 minutes)
pytest tests/unit/ tests/integration/ -v

# Full test suite (5+ minutes)
pytest tests/ -v

# Run specific test type
pytest tests/unit/ -m unit -v
pytest tests/integration/ -m integration -v
pytest tests/e2e/ -m e2e -v
pytest tests/performance/ -m performance -v

# Run with coverage
pytest tests/unit/ --cov=src --cov-report=html
```

## 🎯 Test Focus

Each test type focuses on different aspects:

| Test Type | Focus | Example |
|-----------|-------|---------|
| **Unit** | Function contracts | `test_parse_request_valid_input()` |
| **Integration** | Service interaction | `test_agent_with_llm_service()` |
| **E2E** | User workflows | `test_question_to_marking_workflow()` |
| **Performance** | Load/timing | `test_concurrent_requests()` |

## 🚀 Development Workflow

1. **Write unit tests first** (TDD)
2. **Run unit tests frequently** (every save)
3. **Run integration tests before commit**
4. **Run e2e tests before merge**
5. **Run performance tests before release**

This structure gives you **fast feedback** during development while ensuring **comprehensive coverage** for releases.
