# 🗄️ **Database Usage Guide**

## **Overview**

The Derivativ.ai system currently operates with **two parallel database systems**, each optimized for different use cases. This guide clarifies when to use which system and how they integrate.

---

## **🔄 Parallel Database Architecture**

### **System 1: NeonDBClient (Legacy/Simple)**
- **Purpose**: Basic question generation and storage
- **File**: `src/database/neon_client.py`
- **Used By**: `QuestionGenerationService`, CLI tools, simple generation workflows
- **Tables**: Only `deriv_candidate_questions`
- **Focus**: Fast, simple CRUD operations

### **System 2: DatabaseManager (Advanced/Complete)**
- **Purpose**: Complete session tracking, audit trails, quality control
- **File**: `src/services/database_manager.py`
- **Used By**: `ReActOrchestrator`, `QualityControlWorkflow`, multi-agent pipelines
- **Tables**: All 6 `deriv_*` tables with full schema
- **Focus**: Complete audit trails, quality control integration

---

## **🎯 When to Use Which System**

### **Use NeonDBClient When:**

✅ **Simple Question Generation**
```python
# Basic question generation without quality control
from src.services.generation_service import QuestionGenerationService

service = QuestionGenerationService(database_url)
await service.generate_questions(config)
```

✅ **CLI Operations**
```bash
# Simple CLI commands
python main.py generate --grades 5 --count 3
python main.py list --grade 6
```

✅ **Quick Prototyping**
```python
# Direct database operations
from src.database import NeonDBClient

db = NeonDBClient(database_url)
questions = await db.get_questions_by_grade(5)
```

✅ **Legacy Integration**
- Existing scripts that expect simple database interface
- Backward compatibility requirements
- Single-table operations

### **Use DatabaseManager When:**

✅ **Quality Control Workflows**
```python
# Complete quality control pipeline
from src.services.quality_control_workflow import QualityControlWorkflow

workflow = QualityControlWorkflow(
    review_agent=review_agent,
    refinement_agent=refinement_agent,
    database_manager=database_manager  # ← Uses DatabaseManager
)
```

✅ **Multi-Agent Orchestration**
```python
# Full multi-agent pipeline with audit trails
from src.services.react_orchestrator import ReActMultiAgentOrchestrator

orchestrator = ReActMultiAgentOrchestrator(
    database_manager=database_manager  # ← Uses DatabaseManager
)
```

✅ **Session Tracking Required**
```python
# When you need complete audit trails
session_id = await database_manager.create_session(config_id)
await database_manager.save_llm_interaction(session_id, interaction)
await database_manager.save_question(session_id, question)
```

✅ **Production Deployments**
- Complete audit requirements
- Quality control integration
- Multi-agent coordination
- Error tracking and resolution

---

## **📊 Feature Comparison**

| Feature | NeonDBClient | DatabaseManager |
|---------|--------------|-----------------|
| **Question Storage** | ✅ Basic | ✅ Complete with lineage |
| **Session Tracking** | ❌ None | ✅ Full sessions |
| **LLM Audit Trail** | ❌ None | ✅ Complete interactions |
| **Quality Control** | ❌ Manual | ✅ Automated workflow |
| **Error Tracking** | ❌ Limited | ✅ Comprehensive |
| **Manual Review Queue** | ❌ None | ✅ Integrated |
| **Performance** | ⚡ Fast | 🔄 Comprehensive |
| **Complexity** | 🟢 Simple | 🟡 Advanced |
| **Setup** | 🟢 Minimal | 🟡 Complete schema |

---

## **🏗️ Database Schema Overview**

### **NeonDBClient Tables**
```sql
deriv_candidate_questions (
    -- Basic question data only
    id SERIAL PRIMARY KEY,
    generation_id UUID,
    question_data JSONB,
    -- Minimal tracking fields
)
```

### **DatabaseManager Tables (Complete Schema)**
```sql
-- Session management
deriv_generation_sessions

-- Complete audit trail
deriv_llm_interactions

-- Questions with full lineage
deriv_candidate_questions (extended)

-- Quality control
deriv_review_results
deriv_error_logs
deriv_manual_review_queue
```

---

## **🚀 Migration Patterns**

### **Pattern 1: Gradual Migration**
Start with `NeonDBClient`, upgrade to `DatabaseManager` when needed:

```python
class AdaptiveService:
    def __init__(self, enable_quality_control: bool = False):
        if enable_quality_control:
            self.db = DatabaseManager(database_url)
            self.use_quality_control = True
        else:
            self.db = NeonDBClient(database_url)
            self.use_quality_control = False

    async def generate_question(self, config):
        if self.use_quality_control:
            # Use complete workflow
            return await self.quality_workflow.process_question(...)
        else:
            # Use simple generation
            return await self.generator.generate_question(config)
```

### **Pattern 2: Adapter Pattern**
Use `DatabaseManager` with `NeonDBClient` interface:

```python
class NeonDBClientAdapter:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager
        self._session_cache = {}

    async def save_candidate_question(self, question, session_id=None):
        if not session_id:
            session_id = str(uuid.uuid4())
            # Create session automatically

        return await self.database_manager.save_question(question, session_id)

    async def get_questions_by_grade(self, grade):
        return await self.database_manager.get_questions_by_grade(grade)
```

### **Pattern 3: Unified Interface**
Single service supporting both backends:

```python
class UnifiedQuestionService:
    def __init__(self, backend: str = "simple"):
        if backend == "simple":
            self.db = NeonDBClient(database_url)
        elif backend == "advanced":
            self.db = DatabaseManager(database_url)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    async def generate_questions(self, config, use_quality_control=False):
        if use_quality_control and isinstance(self.db, DatabaseManager):
            # Use quality control workflow
            pass
        else:
            # Use basic generation
            pass
```

---

## **🔧 Configuration Examples**

### **Environment-Based Selection**
```python
import os

def get_database_client():
    if os.getenv("USE_ADVANCED_DB", "false").lower() == "true":
        return DatabaseManager(os.getenv("NEON_DATABASE_URL"))
    else:
        return NeonDBClient(os.getenv("NEON_DATABASE_URL"))
```

### **Feature-Based Selection**
```python
def get_database_for_workflow(workflow_type: str):
    if workflow_type in ["quality_control", "multi_agent", "production"]:
        return DatabaseManager(database_url)
    elif workflow_type in ["simple", "cli", "prototype"]:
        return NeonDBClient(database_url)
    else:
        raise ValueError(f"Unknown workflow type: {workflow_type}")
```

---

## **📋 Best Practices**

### **For Development**

1. **Start Simple**: Use `NeonDBClient` for initial development
2. **Upgrade Gradually**: Move to `DatabaseManager` when you need quality control
3. **Test Both**: Ensure your code works with both systems if needed

### **For Production**

1. **Use DatabaseManager**: For production deployments requiring audit trails
2. **Enable Quality Control**: Always use automated quality workflows
3. **Monitor Performance**: Track database operations and optimize queries

### **For Integration**

1. **Abstract Database Operations**: Don't hardcode database client choice
2. **Support Migration**: Design services to work with both systems
3. **Document Dependencies**: Clearly specify which database features you need

---

## **🎯 Decision Matrix**

Use this matrix to choose the right database system:

| **Need** | **NeonDBClient** | **DatabaseManager** |
|----------|------------------|---------------------|
| Quick prototype | ✅ Perfect | ⚠️ Overkill |
| Simple CLI tool | ✅ Perfect | ⚠️ Overkill |
| Quality control | ❌ Can't do | ✅ Perfect |
| Audit trails | ❌ Can't do | ✅ Perfect |
| Multi-agent workflows | ❌ Limited | ✅ Perfect |
| Manual review queue | ❌ Can't do | ✅ Perfect |
| Error tracking | ❌ Basic only | ✅ Perfect |
| Session management | ❌ None | ✅ Perfect |
| Production deployment | ⚠️ Limited | ✅ Perfect |
| Backward compatibility | ✅ Perfect | ⚠️ Needs adapter |

---

## **🔮 Future Roadmap**

### **Short Term (Recommended)**
- **Maintain Both Systems**: They serve different purposes well
- **Document Usage Patterns**: Clear guidelines for when to use which
- **Create Adapter Utilities**: Easy migration between systems

### **Medium Term (Optional)**
- **Unified Interface**: Single API supporting both backends
- **Feature Flags**: Runtime switching between simple/advanced modes
- **Performance Optimization**: Optimize both systems for their use cases

### **Long Term (Future)**
- **Single System**: Eventually consolidate if clear benefits emerge
- **Advanced Features**: Add more sophisticated database capabilities
- **Cloud Integration**: Better integration with cloud database services

---

## **📞 Support**

### **When to Use Each System**

**❓ I want to generate a few questions quickly**
→ Use `NeonDBClient` with `QuestionGenerationService`

**❓ I need complete audit trails and quality control**
→ Use `DatabaseManager` with `QualityControlWorkflow`

**❓ I'm building a production system**
→ Use `DatabaseManager` for full capabilities

**❓ I'm prototyping or testing**
→ Use `NeonDBClient` for simplicity

**❓ I need to track who approved/rejected questions**
→ Use `DatabaseManager` for manual review queue

**❓ I want to monitor LLM performance and costs**
→ Use `DatabaseManager` for interaction tracking

### **Migration Help**

If you need to migrate from one system to another, refer to the migration patterns above or check the `DATABASE_CONSOLIDATION_PLAN.md` for detailed migration strategies.

---

**💡 Remember**: Both systems use the same `deriv_*` table naming convention, so data compatibility is maintained. The choice is about features and complexity, not data format.
