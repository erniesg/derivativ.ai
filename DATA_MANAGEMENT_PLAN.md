# 📊 **Data Management & Quality Control Plan**

## **Overview**

This document outlines the comprehensive data management strategy for the Cambridge IGCSE Mathematics Multi-Agent Question Generation System, addressing:

1. **🚦 Automated quality control criteria**
2. **💾 Complete data storage architecture**
3. **🔄 LLM interaction logging**
4. **🔗 Data lineage and traceability**
5. **⚠️ Error tracking and debugging**

---

## **1. 🚦 Automated Insertion Criteria**

### **Quality Thresholds**

```python
class InsertionCriteria:
    # Score thresholds (0.0 to 1.0)
    AUTO_APPROVE_THRESHOLD = 0.85      # Auto-insert if score ≥ 0.85
    MANUAL_REVIEW_THRESHOLD = 0.70     # Manual review if 0.70 ≤ score < 0.85
    AUTO_REJECT_THRESHOLD = 0.70       # Auto-reject if score < 0.70

    # Component minimums
    MIN_SYLLABUS_COMPLIANCE = 0.80     # Must align with curriculum
    MIN_MARKING_QUALITY = 0.75         # Cambridge marking standards
    MIN_DIFFICULTY_ALIGNMENT = 0.65    # Grade-appropriate complexity

    # Outcome rules
    FORBIDDEN_OUTCOMES = [ReviewOutcome.REJECT]
    MANUAL_OUTCOMES = [ReviewOutcome.MAJOR_REVISIONS]
```

### **Decision Logic**

1. **Auto-Reject**: `ReviewOutcome.REJECT` OR `overall_score < 0.70`
2. **Manual Review**: Component scores below minimum OR `MAJOR_REVISIONS` outcome
3. **Auto-Approve**: `overall_score ≥ 0.85` AND all components above minimums
4. **Default**: Manual review for edge cases

### **Insertion Status Types**

- `AUTO_APPROVED` → Immediate database insertion
- `AUTO_REJECTED` → Discard, log for analysis
- `MANUAL_REVIEW` → Queue for human review
- `MANUALLY_APPROVED` → Human-approved insertion
- `MANUALLY_REJECTED` → Human-rejected with feedback

---

## **2. 📊 Complete Data Storage Architecture**

### **Database Schema**

```sql
-- Generation Sessions (top-level tracking)
deriv_generation_sessions (
    session_id UUID PRIMARY KEY,
    config_id VARCHAR NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    status VARCHAR CHECK (status IN ('running', 'completed', 'failed')),
    total_questions_requested INT NOT NULL,
    questions_generated INT DEFAULT 0,
    questions_approved INT DEFAULT 0,
    error_count INT DEFAULT 0,
    summary_metrics JSONB  -- Average scores, success rates, etc.
)

-- Raw LLM Interactions (complete audit trail)
deriv_llm_interactions (
    interaction_id UUID PRIMARY KEY,
    session_id UUID REFERENCES deriv_generation_sessions,
    agent_type VARCHAR NOT NULL,  -- 'generator', 'marker', 'reviewer', 'refiner'
    model_used VARCHAR NOT NULL,  -- Model identifier
    prompt_text TEXT NOT NULL,    -- Complete input prompt
    raw_response TEXT,            -- Raw LLM output
    parsed_response JSONB,        -- Structured parsed data
    processing_time_ms INT,       -- Performance metrics
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    timestamp TIMESTAMP DEFAULT NOW(),

    -- Additional metadata
    temperature FLOAT,
    max_tokens INT,
    token_usage JSONB  -- Input/output token counts
)

-- Candidate Questions (with full lineage tracking)
deriv_candidate_questions (
    -- All existing CandidateQuestion fields PLUS:
    question_id UUID PRIMARY KEY,
    session_id UUID REFERENCES deriv_generation_sessions,
    generation_interaction_id UUID REFERENCES deriv_llm_interactions,
    marking_interaction_id UUID REFERENCES deriv_llm_interactions,
    review_interaction_id UUID REFERENCES deriv_llm_interactions,

    -- Question content (stored as JSONB for flexibility)
    question_data JSONB NOT NULL,

    -- Educational metadata (extracted for indexing)
    subject_content_refs TEXT[],
    topic_path TEXT[],
    command_word VARCHAR(50),
    target_grade INT CHECK (target_grade BETWEEN 1 AND 12),
    marks INT CHECK (marks > 0),
    calculator_policy VARCHAR(20) CHECK (calculator_policy IN ('allowed', 'not_allowed', 'assumed')),
    curriculum_type VARCHAR(50) DEFAULT 'cambridge_igcse',

    -- Quality control
    insertion_status VARCHAR(30) CHECK (insertion_status IN (
        'pending', 'auto_approved', 'manual_review',
        'auto_rejected', 'manually_approved', 'manually_rejected',
        'needs_revision', 'archived'
    )) DEFAULT 'pending',

    -- Validation results
    validation_passed BOOLEAN,
    validation_warnings INT DEFAULT 0,
    validation_errors JSONB,

    -- Review workflow
    review_score DECIMAL(3,2) CHECK (review_score BETWEEN 0 AND 1),
    insertion_timestamp TIMESTAMP,
    approved_by VARCHAR(100),
    rejection_reason TEXT,
    manual_notes TEXT,

    -- Version control
    version INT DEFAULT 1,
    parent_question_id UUID REFERENCES deriv_candidate_questions(question_id),

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
)

-- Review Results (quality assessment data)
deriv_review_results (
    review_id UUID PRIMARY KEY,
    question_id UUID REFERENCES deriv_candidate_questions,
    interaction_id UUID REFERENCES deriv_llm_interactions,

    -- Review outcomes
    outcome VARCHAR NOT NULL,  -- ReviewOutcome enum values
    overall_score DECIMAL(3,2) CHECK (overall_score BETWEEN 0 AND 1),

    -- Component scores
    mathematical_accuracy DECIMAL(3,2) CHECK (mathematical_accuracy BETWEEN 0 AND 1),
    syllabus_compliance DECIMAL(3,2) CHECK (syllabus_compliance BETWEEN 0 AND 1),
    difficulty_alignment DECIMAL(3,2) CHECK (difficulty_alignment BETWEEN 0 AND 1),
    marking_quality DECIMAL(3,2) CHECK (marking_quality BETWEEN 0 AND 1),
    pedagogical_soundness DECIMAL(3,2) CHECK (pedagogical_soundness BETWEEN 0 AND 1),
    technical_quality DECIMAL(3,2) CHECK (technical_quality BETWEEN 0 AND 1),

    -- Detailed feedback
    feedback_summary TEXT,
    specific_feedback JSONB,      -- Structured detailed feedback
    suggested_improvements JSONB, -- Array of improvement suggestions

    timestamp TIMESTAMP DEFAULT NOW()
)

-- Comprehensive Error Logs
deriv_error_logs (
    error_id UUID PRIMARY KEY,
    session_id UUID REFERENCES deriv_generation_sessions,
    interaction_id UUID REFERENCES deriv_llm_interactions,
    question_id UUID REFERENCES deriv_candidate_questions,

    -- Error classification
    error_type VARCHAR NOT NULL,  -- 'parsing', 'model_call', 'validation', etc.
    error_severity VARCHAR CHECK (error_severity IN ('low', 'medium', 'high', 'critical')),

    -- Error details
    error_message TEXT NOT NULL,
    stack_trace TEXT,
    context_data JSONB,          -- Additional context for debugging

    -- Resolution tracking
    resolved BOOLEAN DEFAULT FALSE,
    resolution_notes TEXT,
    resolved_by VARCHAR,
    resolved_at TIMESTAMP,

    timestamp TIMESTAMP DEFAULT NOW()
)

-- Manual Review Queue (for human reviewers)
deriv_manual_review_queue (
    queue_id UUID PRIMARY KEY,
    question_id UUID REFERENCES deriv_candidate_questions,
    review_id UUID REFERENCES deriv_review_results,

    -- Queue management
    priority INT DEFAULT 1,      -- 1=low, 5=high
    assigned_to VARCHAR,         -- Reviewer user ID
    status VARCHAR DEFAULT 'pending' CHECK (status IN ('pending', 'assigned', 'in_review', 'completed', 'escalated')),

    -- Review tracking
    review_started_at TIMESTAMP,
    review_completed_at TIMESTAMP,
    estimated_time_minutes INT,
    actual_time_minutes INT,
    reviewer_notes TEXT,
    final_decision VARCHAR,      -- 'approved', 'rejected', 'needs_revision', 'escalated'

    created_at TIMESTAMP DEFAULT NOW()
)
```

### **Data Relationships**

```
GenerationSession
├── LLMInteraction[] (all agent calls)
├── CandidateQuestion[] (generated questions)
├── ReviewResult[] (quality assessments)
├── ErrorLog[] (any failures)
└── ManualReviewQueue[] (human review items)

CandidateQuestion
├── generation_interaction → LLMInteraction (how it was generated)
├── marking_interaction → LLMInteraction (marking scheme creation)
├── review_interaction → LLMInteraction (quality review)
└── review_result → ReviewResult (quality scores/feedback)
```

---

## **3. 🔄 Raw LLM Data Capture**

### **What We Store**

#### **Inputs (Prompts)**
```python
{
    "prompt_text": "Complete formatted prompt sent to LLM",
    "model_used": "gpt-4o-mini / claude-sonnet-4 / etc",
    "parameters": {
        "temperature": 0.7,
        "max_tokens": 2000,
        "system_prompt": "...",
        "user_prompt": "..."
    },
    "context_data": {
        "config_id": "mixed_review_gpt4o_mini",
        "target_grade": 5,
        "subject_refs": ["A1.1", "A1.2"]
    }
}
```

#### **Outputs (Responses)**
```python
{
    "raw_response": "Complete unprocessed LLM output",
    "parsed_response": {
        "structured_data": "Parsed JSON/objects",
        "extraction_method": "json_block / regex / manual",
        "parsing_success": true
    },
    "metadata": {
        "processing_time_ms": 2340,
        "token_usage": {
            "input_tokens": 1500,
            "output_tokens": 800,
            "total_cost_usd": 0.023
        },
        "response_quality": {
            "valid_json": true,
            "complete_response": true,
            "truncated": false
        }
    }
}
```

#### **Error Information**
```python
{
    "error_type": "parsing_failure",
    "error_message": "JSON extraction failed",
    "stack_trace": "Full Python traceback",
    "context": {
        "agent_type": "reviewer",
        "question_id": "Q_12345",
        "config_id": "mixed_review_gpt4o_mini",
        "retry_attempt": 2,
        "fallback_used": true
    },
    "recovery_action": "used_fallback_parsing"
}
```

---

## **4. 🔗 Data Lineage & Traceability**

### **Question Lineage Tracking**

Every question has a complete audit trail:

```python
class QuestionLineage:
    # Origin
    session_id: UUID
    config_id: str
    generation_timestamp: datetime

    # Generation chain
    generation_interaction_id: UUID  # How question was created
    marking_interaction_id: UUID     # How marking scheme was created
    review_interaction_id: UUID      # How quality was assessed

    # All related data
    prompts_used: List[str]          # All prompts in pipeline
    models_used: List[str]           # All models in pipeline
    processing_times: List[int]      # Performance at each step

    # Quality control
    review_scores: ReviewFeedback    # Complete quality assessment
    insertion_decision: InsertionStatus
    approval_trail: List[ApprovalEvent]
```

### **Cross-Reference Queries**

```sql
-- Find all questions from a session
SELECT q.*, r.overall_score, r.outcome
FROM deriv_candidate_questions q
JOIN deriv_review_results r ON q.question_id = r.question_id
WHERE q.session_id = ?

-- Trace a question's complete pipeline
SELECT
    i.agent_type,
    i.model_used,
    i.success,
    i.processing_time_ms,
    i.error_message
FROM deriv_llm_interactions i
WHERE i.interaction_id IN (
    SELECT unnest(ARRAY[
        q.generation_interaction_id,
        q.marking_interaction_id,
        q.review_interaction_id
    ])
    FROM deriv_candidate_questions q
    WHERE q.question_id = ?
)

-- Find questions needing manual review
SELECT q.*, r.feedback_summary, r.suggested_improvements
FROM deriv_candidate_questions q
JOIN deriv_review_results r ON q.question_id = r.question_id
WHERE q.insertion_status = 'manual_review'
ORDER BY r.overall_score DESC
```

---

## **5. ⚠️ Error Tracking & Debugging**

### **Error Classification**

```python
class ErrorType(Enum):
    MODEL_CALL_FAILED = "model_call_failed"       # LLM API failures
    PARSING_FAILED = "parsing_failed"             # JSON/response parsing
    VALIDATION_FAILED = "validation_failed"       # Schema validation
    TIMEOUT = "timeout"                           # Processing timeouts
    RATE_LIMIT = "rate_limit"                     # API rate limiting
    CONTENT_FILTER = "content_filter"             # Content policy violations
    CONFIG_ERROR = "config_error"                 # Configuration issues
    DATABASE_ERROR = "database_error"             # DB operations
    UNKNOWN = "unknown"                           # Unclassified errors

class ErrorSeverity(Enum):
    LOW = "low"          # Non-blocking, fallback available
    MEDIUM = "medium"    # Affects quality, manual intervention helpful
    HIGH = "high"        # Blocks processing, requires fix
    CRITICAL = "critical" # System-wide issue, immediate attention
```

### **Error Recovery Strategies**

```python
# Automatic recovery actions
RECOVERY_STRATEGIES = {
    "parsing_failed": [
        "retry_with_different_extraction",
        "use_fallback_parser",
        "request_simplified_format"
    ],
    "model_call_failed": [
        "retry_with_exponential_backoff",
        "switch_to_backup_model",
        "reduce_complexity"
    ],
    "rate_limit": [
        "implement_backoff",
        "queue_for_later",
        "switch_provider"
    ]
}
```

### **Debug Information Capture**

For every error, we capture:

1. **Complete Context**: Session, config, question data
2. **Timing Information**: When error occurred in pipeline
3. **Environment State**: Model availability, rate limits
4. **User Actions**: What triggered the generation
5. **Recovery Attempts**: What was tried to fix it
6. **Final Resolution**: How it was ultimately resolved

---

## **6. 🎯 Implementation Status**

### **✅ Completed**

1. **DatabaseManager**: Complete schema with all 6 `deriv_*` tables
2. **QualityControlWorkflow**: Complete automated quality improvement loop
3. **RefinementAgent**: Question refinement with fallback strategies
4. **MultiAgentOrchestrator**: Complete pipeline coordination with audit trails
5. **InsertionCriteria**: Automated decision making with configurable thresholds
6. **LLMInteraction**: Raw data capture framework with performance metrics
7. **GenerationSession**: Session management with complete audit trails
8. **Error Logging**: Comprehensive error tracking and resolution workflow
9. **Review Integration**: Complete ReviewAgent integration with quality scoring

### **🚧 Current Systems**

#### **Production Ready Systems**
1. **DatabaseManager**: All 6 tables, complete audit trails, used by ReActOrchestrator
2. **NeonDBClient**: Single table, simple CRUD, used by GenerationService
3. **Quality Control**: Complete automated refinement and decision workflow
4. **Agent Architecture**: Generator, Marker, Reviewer, Refinement agents

#### **Integration Status**
- **ReActOrchestrator**: Uses DatabaseManager for complete session tracking
- **GenerationService**: Uses NeonDBClient for basic question storage
- **Quality Workflow**: Integrated with DatabaseManager for audit trails
- **Manual Review**: Queue system integrated with quality control decisions

### **📋 Usage Example**

```python
# Complete Quality Control Workflow
from src.services.quality_control_workflow import QualityControlWorkflow

# Initialize with all agents
workflow = QualityControlWorkflow(
    review_agent=review_agent,
    refinement_agent=refinement_agent,
    generator_agent=generator_agent,
    database_manager=database_manager,
    auto_publish=True
)

# Process question through complete workflow
result = await workflow.process_question(
    question=candidate_question,
    session_id=session_id,
    generation_config=config
)

# Result contains:
# - Final decision (approve/refine/reject/manual_review)
# - Complete audit trail of all steps
# - Auto-publication status if enabled
# - Performance metrics and timing data

print(f"Final decision: {result['final_decision']}")
print(f"Total iterations: {result['total_iterations']}")
print(f"Auto-published: {result.get('payload_question_id') is not None}")
```

This system provides **complete transparency, traceability, and quality control** for the question generation pipeline, enabling both automated operation and easy debugging/improvement with full database consistency using the `deriv_*` table naming convention.
