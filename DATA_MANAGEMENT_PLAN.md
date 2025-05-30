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
generation_sessions (
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
llm_interactions (
    interaction_id UUID PRIMARY KEY,
    session_id UUID REFERENCES generation_sessions,
    agent_type VARCHAR NOT NULL,  -- 'generator', 'marker', 'reviewer'
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

-- Extended Candidate Questions (with full lineage)
candidate_questions_extended (
    -- All existing CandidateQuestion fields PLUS:
    session_id UUID REFERENCES generation_sessions,
    generation_interaction_id UUID REFERENCES llm_interactions,
    marking_interaction_id UUID REFERENCES llm_interactions,
    review_interaction_id UUID REFERENCES llm_interactions,

    -- Quality control
    insertion_status VARCHAR CHECK (insertion_status IN (
        'pending', 'auto_approved', 'manual_review',
        'auto_rejected', 'manually_approved', 'manually_rejected'
    )),
    insertion_timestamp TIMESTAMP,
    approved_by VARCHAR,  -- 'system' or user_id
    rejection_reason TEXT,

    -- Audit trail
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    version INT DEFAULT 1
)

-- Review Results (quality assessment data)
review_results (
    review_id UUID PRIMARY KEY,
    question_id UUID REFERENCES candidate_questions_extended,
    interaction_id UUID REFERENCES llm_interactions,

    -- Review outcomes
    outcome VARCHAR NOT NULL,  -- ReviewOutcome enum values
    overall_score DECIMAL(3,2) CHECK (overall_score BETWEEN 0 AND 1),

    -- Component scores
    syllabus_compliance DECIMAL(3,2) CHECK (syllabus_compliance BETWEEN 0 AND 1),
    difficulty_alignment DECIMAL(3,2) CHECK (difficulty_alignment BETWEEN 0 AND 1),
    marking_quality DECIMAL(3,2) CHECK (marking_quality BETWEEN 0 AND 1),

    -- Detailed feedback
    feedback_summary TEXT,
    specific_feedback JSONB,      -- Structured detailed feedback
    suggested_improvements JSONB, -- Array of improvement suggestions

    timestamp TIMESTAMP DEFAULT NOW()
)

-- Comprehensive Error Logs
error_logs (
    error_id UUID PRIMARY KEY,
    session_id UUID REFERENCES generation_sessions,
    interaction_id UUID REFERENCES llm_interactions,

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
manual_review_queue (
    queue_id UUID PRIMARY KEY,
    question_id UUID REFERENCES candidate_questions_extended,
    review_id UUID REFERENCES review_results,

    -- Queue management
    priority INT DEFAULT 1,      -- 1=low, 5=high
    assigned_to VARCHAR,         -- Reviewer user ID
    status VARCHAR DEFAULT 'pending' CHECK (status IN ('pending', 'in_review', 'completed')),

    -- Review tracking
    review_started_at TIMESTAMP,
    review_completed_at TIMESTAMP,
    reviewer_notes TEXT,
    final_decision VARCHAR,      -- 'approve', 'reject', 'revise'

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
FROM candidate_questions_extended q
JOIN review_results r ON q.question_id = r.question_id
WHERE q.session_id = ?

-- Trace a question's complete pipeline
SELECT
    i.agent_type,
    i.model_used,
    i.success,
    i.processing_time_ms,
    i.error_message
FROM llm_interactions i
WHERE i.interaction_id IN (
    SELECT unnest(ARRAY[
        q.generation_interaction_id,
        q.marking_interaction_id,
        q.review_interaction_id
    ])
    FROM candidate_questions_extended q
    WHERE q.question_id = ?
)

-- Find questions needing manual review
SELECT q.*, r.feedback_summary, r.suggested_improvements
FROM candidate_questions_extended q
JOIN review_results r ON q.question_id = r.question_id
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

1. **MultiAgentOrchestrator**: Complete pipeline coordination
2. **InsertionCriteria**: Automated decision making
3. **LLMInteraction**: Raw data capture framework
4. **GenerationSession**: Session management with audit trails
5. **Error Logging**: Comprehensive error tracking
6. **Quality Control**: ReviewAgent integration

### **🚧 Next Steps**

1. **Database Implementation**: Implement actual Neon DB schemas
2. **Agent Enhancement**: Modify agents to return raw prompts/responses
3. **Manual Review Interface**: Build UI for human reviewers
4. **Analytics Dashboard**: Session/error analysis tools
5. **Cost Tracking**: Monitor API usage and costs
6. **Performance Optimization**: Batch processing, caching

### **📋 Usage Example**

```python
# Initialize orchestrator
orchestrator = MultiAgentOrchestrator(
    generator_model=gpt4o_mini,
    marker_model=gpt4o_mini,
    reviewer_model=claude_sonnet_4,
    db_client=neon_db_client,
    debug=True
)

# Generate questions with full quality control
session = await orchestrator.generate_questions_with_quality_control(
    config_id="mixed_review_gpt4o_mini",
    num_questions=10,
    auto_insert=True  # Auto-approve high-quality questions
)

# Get comprehensive session analytics
summary = orchestrator.get_session_summary(session)
print(f"Success rate: {summary['questions']['success_rate']:.1%}")
print(f"Average quality: {summary['quality_metrics']['average_score']:.2f}")
print(f"Auto-approved: {summary['insertion_decisions']['auto_approved']}")
print(f"Manual review needed: {summary['insertion_decisions']['manual_review']}")
```

This system provides **complete transparency, traceability, and quality control** for the question generation pipeline, enabling both automated operation and easy debugging/improvement.
