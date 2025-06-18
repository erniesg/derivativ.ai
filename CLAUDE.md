# CLAUDE.md - Derivativ AI Education Platform

## 🎯 PROJECT MISSION & HACKATHON GOALS

**Project Name**: Derivativ - AI-Powered Math Tutor
**Primary Mission**: Create a production-ready AI education platform that generates Cambridge IGCSE Mathematics questions using sophisticated multi-agent coordination with real-time quality control.

**Hackathon Goal (5 days)**: Build an impressive, fully-functional system that demonstrates advanced AI agent workflows and real educational value for teachers and students.

**Target Users**: 
- **Teachers**: Generate high-quality assessment materials efficiently
- **Students**: Personalized learning with adaptive difficulty progression
- **Future Vision**: Interactive video generation with student context and cultural adaptation

---

## 🏗️ SYSTEM ARCHITECTURE & DESIGN PATTERNS

### Core Design Philosophy
- **Multi-Agent Coordination**: Specialized AI agents (Generator, Marker, Reviewer, Refiner) work together with visible reasoning
- **Test-Driven Development**: Comprehensive test suite written before implementation for reliability
- **Quality-First**: Automatic quality assessment and improvement cycles
- **Real-Time Transparency**: Agent decision-making visible to users and judges
- **Production Architecture**: Built for scale and real-world deployment from day one

### Multi-Agent Coordination Patterns

#### 1. **smolagents Multi-Agent Pattern** (`src/derivativ/agents/orchestrator.py`)
```
Request → Modal Function → smolagents → [QuestionAgent, MarkerAgent, ReviewAgent, RefinerAgent] → Response
- smolagents framework handles agent coordination and communication
- Each agent deployed as Modal function for scalable compute
- Reasoning steps logged via smolagents built-in tracking
- Automatic agent selection based on task requirements
```

#### 2. **Modal + Neon DB Integration** (`src/derivativ/database/neon_client.py`)
```
Modal Function → Neon DB Connection → Agent Data Persistence → Audit Trails
- Serverless PostgreSQL with automatic scaling
- Full audit trails for all agent interactions
- Connection pooling optimized for Modal's execution model
- Real-time data synchronization across agent instances
```

#### 3. **Cloudflare Edge Computing** (`cloudflare-workers/api-gateway.js`)
```
Frontend → Cloudflare Workers → Modal Agent Functions → Neon DB
- Edge API gateway for global low-latency access
- Intelligent request routing based on geography and load
- Caching layer with R2 for frequently requested content
- Real-time streaming via WebSockets at the edge
```

#### 4. **AI SDK Streaming Components** (`frontend/components/AgentReasoning.tsx`)
```
Modal Agent Updates → Cloudflare Workers → AI SDK → Streaming React Components
- Real-time agent reasoning display using AI SDK
- Streaming components update as agents make decisions
- Live progress indicators for multi-agent workflows
- Demo-friendly transparent decision-making
```

### State Management & Persistence
- **Neon DB**: Serverless PostgreSQL for agent data, audit trails, and question storage
- **Cloudflare D1**: SQLite edge database for CMS content and caching
- **Modal State**: Distributed agent state management across compute instances
- **smolagents Logging**: Built-in reasoning and decision tracking
- **Cloudflare R2**: Object storage for static assets and large content files

---

## 🛠️ DERIVATIV TECHNOLOGY STACK

### **Production Technology Decisions**

#### Database & Storage Layer
- **Neon DB**: Serverless PostgreSQL for past papers, syllabus, command words, and candidate questions
- **Cloudflare D1**: SQLite-based database for CMS content and caching
- **Cloudflare R2**: Object storage for static assets and media files

#### AI Agent Infrastructure
- **smolagents**: Facebook's production-ready multi-agent framework
- **Modal**: Serverless compute platform for AI agent execution
- **LLM Providers**: OpenAI, Anthropic, Google Gemini, OpenRouter (selectable per agent/step)
- **Model Routing**: Dynamic provider selection based on task complexity and cost

#### Content Management & API
- **Payload CMS**: Headless CMS deployed on Cloudflare for content management
- **Cloudflare Workers**: Edge computing for API gateway and orchestration (future)
- **FastAPI**: Core API framework for agent coordination and question generation

#### Frontend & User Experience
- **Next.js**: React framework with AI SDK for streaming components
- **TypeScript**: Type safety across full stack
- **Tailwind CSS**: Rapid styling with professional appearance
- **AI SDK**: Streaming React components for real-time agent reasoning display

#### Development & Testing
- **pytest**: Comprehensive async testing framework
- **pytest-cov**: Code coverage reporting (targeting >95%)
- **smolagents testing**: Agent coordination and workflow testing
- **Cloudflare Workers testing**: Edge function testing tools

#### Future Enhancements
- **Manim**: Mathematical animation and diagram generation
- **Cloudflare Durable Objects**: Stateful edge computing for complex workflows
- **Cloudflare Analytics**: Performance monitoring and usage tracking

### **Architecture Benefits**

#### Serverless & Edge-First
- **Modal serverless compute** for automatic scaling and cost optimization
- **Cloudflare edge network** for global low-latency access
- **Neon DB serverless PostgreSQL** with automatic scaling and branching
- **Zero cold-start architecture** for consistent performance

#### Production Sophistication
- **smolagents framework** for robust multi-agent coordination
- **Edge caching with R2** for optimal content delivery
- **Real-time streaming** via AI SDK and WebSockets
- **Global distribution** through Cloudflare's network

#### Demo & Development Appeal
- **Live agent reasoning** streamed to judges in real-time
- **Professional edge infrastructure** showing production readiness
- **Multiple LLM providers** with intelligent routing
- **Transparent decision-making** via smolagents logging

---

## 📋 DERIVATIV IMPLEMENTATION ROADMAP (5 Days)

### **Day 1: Foundation & Core Systems** (8 hours)

#### Project Setup & Infrastructure
- [ ] **Initialize Derivativ project structure**: Modal + Neon DB + Cloudflare integration
- [ ] **Setup smolagents framework**: Agent coordination and communication patterns
- [ ] **Configure Neon DB connection**: Database schema for questions, sessions, audit trails
- [ ] **Implement Pydantic models**: Question, GenerationRequest, AgentResult with validation

#### Modal Agent Deployment & Testing
- [ ] **Create Modal agent functions**: QuestionGenerator, Marker, Review, Refiner agents
- [ ] **Write comprehensive test suite**: Agent coordination and smolagents integration tests
- [ ] **LLM provider integration**: OpenAI, Anthropic, Gemini with OpenRouter routing
- [ ] **Database persistence layer**: Neon DB operations with connection pooling

### **Day 2: smolagents Multi-Agent Coordination** (8 hours)

#### Agent Implementation with smolagents
- [ ] **QuestionGeneratorAgent**: Modal function using smolagents for math question creation
- [ ] **MarkerAgent**: Marking scheme generation with Cambridge IGCSE compliance
- [ ] **ReviewAgent**: Quality assessment using smolagents decision-making framework
- [ ] **RefinementAgent**: Question improvement based on review feedback loops

#### Modal + smolagents Integration
- [ ] **Orchestrator deployment**: Main Modal function coordinating all agents via smolagents
- [ ] **Quality control workflow**: Automatic decision-making with smolagents reasoning
- [ ] **Agent communication**: Inter-agent messaging and state sharing through smolagents
- [ ] **Real-time logging**: Agent reasoning steps tracked and streamed to frontend

### **Day 3: Cloudflare Workers + Next.js Frontend** (8 hours)

#### Cloudflare Workers API Gateway
- [ ] **Edge API endpoints**: Routes to Modal agent functions with global distribution
- [ ] **WebSocket handling**: Real-time agent updates streamed from Modal to frontend
- [ ] **Caching strategy**: R2 object storage for static content and frequently requested data
- [ ] **Request routing**: Intelligent load balancing and geographic optimization

#### Next.js + AI SDK Frontend
- [ ] **Streaming components**: Real-time agent reasoning display using AI SDK
- [ ] **Question generation interface**: Professional form integrated with Cloudflare Workers
- [ ] **Live progress indicators**: Agent activity streamed via AI SDK streaming components
- [ ] **Payload CMS integration**: Content management through Cloudflare-hosted Payload

### **Day 4: Polish & Cambridge Integration** (8 hours)

#### Cambridge IGCSE Compliance
- [ ] **Curriculum validation**: Implement Cambridge subject content reference validation
- [ ] **Command word integration**: Proper usage of Cambridge assessment terminology
- [ ] **Grade-appropriate difficulty**: Automatic difficulty validation for target grades
- [ ] **Quality benchmarking**: Mathematical correctness and curriculum alignment scoring

#### Performance & Reliability
- [ ] **Sub-30-second generation**: Optimize for fast demo performance with caching
- [ ] **Concurrent request handling**: Test system stability under multiple simultaneous users
- [ ] **Error resilience**: Graceful fallback strategies for API failures and network issues
- [ ] **Demo preparation**: Pre-warmed scenarios and backup cached responses

### **Day 5: Final Integration & Hackathon Prep** (8 hours)

#### End-to-End Testing & Deployment
- [ ] **Comprehensive system testing**: All workflows work reliably for live demo
- [ ] **Docker deployment setup**: Consistent environment with clear installation docs
- [ ] **Performance validation**: Confirm sub-30-second generation consistently
- [ ] **Demo rehearsal**: Practice 5-minute presentation flow multiple times

#### Hackathon Presentation Preparation
- [ ] **Demo script creation**: Polished narrative showcasing multi-agent coordination
- [ ] **Backup strategies**: Offline mode, cached responses, pre-recorded fallbacks
- [ ] **Technical showcase**: Architecture diagrams highlighting production readiness
- [ ] **Impact demonstration**: Clear value proposition for teachers and students

---

## 🧪 TESTING STRATEGY (TDD Approach)

### Test-First Development Philosophy

#### 1. **Core Workflow Tests** (Write First - Day 1)
```python
# Critical demo functionality
test_generate_question_end_to_end()           # Main demo flow
test_multi_agent_coordination_visible()       # Agent reasoning display
test_quality_control_improvement_cycle()      # Automatic refinement
test_real_time_progress_tracking()            # Live updates

# API integration tests
test_api_question_generation_endpoint()       # REST API functionality
test_api_real_time_websocket_updates()        # Live progress updates
test_api_concurrent_request_handling()        # Scale simulation
```

#### 2. **Agent Coordination Tests** (Day 2)
```python
# Individual agent functionality
test_question_generator_agent()               # Valid question creation
test_marker_agent_scheme_generation()         # Proper mark allocation
test_review_agent_quality_scoring()           # Consistent assessment
test_refinement_agent_improvements()          # Meaningful enhancements

# Multi-agent workflows
test_orchestrator_agent_coordination()        # End-to-end workflow
test_quality_control_workflow_decisions()     # Automatic decisions
```

#### 3. **Performance & Reliability Tests** (Day 4-5)
```python
# Demo requirements
test_generation_completes_under_30_seconds()  # Speed requirement
test_concurrent_generation_stability()        # Multiple users
test_demo_scenario_preloading()               # Fast startup
test_llm_provider_fallback_switching()        # Network resilience
```

### TDD Success Metrics

#### Test Coverage Goals
- [ ] **>95% line coverage** on core business logic
- [ ] **100% coverage** on agent coordination workflows
- [ ] **Demo-critical paths** fully tested with multiple scenarios
- [ ] **Performance benchmarks** embedded in automated tests

#### Quality Assurance Criteria
- [ ] **Mathematical accuracy**: Generated solutions are mathematically correct
- [ ] **Cambridge compliance**: All content references validate against official syllabus
- [ ] **Grade appropriateness**: Difficulty matches target grade level consistently
- [ ] **Agent reasoning quality**: Decision-making logic is sound and visible

#### Demo Reliability Standards
- [ ] **Sub-30-second generation**: Consistently meets speed requirements
- [ ] **Graceful error handling**: No crashes during live presentation
- [ ] **Fallback strategies**: Multiple layers of failure recovery
- [ ] **Real-time updates**: WebSocket connections stable and responsive

### Demo Testing Protocol

#### Pre-Demo Validation
- [ ] **LLM connectivity**: OpenAI and Anthropic APIs accessible with valid keys
- [ ] **Database functionality**: PostgreSQL connection stable and responsive  
- [ ] **Frontend responsiveness**: React app loads quickly with all components
- [ ] **WebSocket stability**: Real-time updates working consistently

#### Live Demo Requirements
- [ ] **Generation speed**: Complete workflow finishes in under 30 seconds
- [ ] **Agent reasoning visibility**: All decision-making steps clearly displayed
- [ ] **Quality metrics display**: Scores and feedback prominently shown
- [ ] **Error recovery**: Graceful fallback to cached examples if needed

---

## 🎖️ DERIVATIV DEVELOPMENT BEST PRACTICES

### Agent Architecture Standards

#### Base Agent Pattern (TDD Implementation)
```python
# src/derivativ/agents/base.py
class BaseAgent(ABC):
    def __init__(self, llm_interface, name: str, config: Dict = None):
        self.llm = llm_interface
        self.name = name
        self.config = config or {}
        self.reasoning_steps = []
        
    def process(self, input_data: Dict) -> AgentResult:
        try:
            self._observe(f"Processing request: {input_data}")
            self._think("Analyzing requirements and planning approach")
            
            output = self._execute(input_data)
            
            return AgentResult(
                success=True,
                agent_name=self.name,
                output=output,
                reasoning_steps=self.reasoning_steps.copy(),
                processing_time=self._get_timing()
            )
        except Exception as e:
            return self._handle_error(e)
    
    @abstractmethod
    def _execute(self, input_data: Dict) -> Dict:
        """Implement agent-specific logic"""
        pass
```

#### Error Handling & Resilience
```python
# Multi-layer fallback strategy
async def generate_with_fallback(self, request: GenerationRequest):
    providers = ["openai", "anthropic", "cached_examples"]
    
    for provider in providers:
        try:
            result = await self._try_provider(provider, request)
            if result.success:
                return result
        except Exception as e:
            self._log_error(f"{provider} failed: {e}")
            continue
    
    return self._emergency_response()
```

### Multi-Agent Coordination Principles

#### Reasoning Transparency (Demo-Critical)
```python
# Every agent logs decision-making for visibility
def _observe(self, observation: str, data: Any = None):
    self.reasoning_steps.append({
        "type": "observation",
        "content": observation,
        "timestamp": time.time(),
        "data": data
    })

def _think(self, reasoning: str):
    self.reasoning_steps.append({
        "type": "thought", 
        "content": reasoning,
        "timestamp": time.time()
    })

def _act(self, action: str, result: Any = None):
    self.reasoning_steps.append({
        "type": "action",
        "content": action,
        "timestamp": time.time(),
        "result": result
    })
```

#### Quality Control Integration
```python
# Automatic quality decisions with clear reasoning
class QualityControlWorkflow:
    thresholds = {
        "auto_approve": 0.85,    # High quality → immediate approval
        "manual_review": 0.70,   # Good quality → human review queue  
        "refine": 0.50,          # Medium quality → improvement cycle
        "regenerate": 0.30,      # Low quality → try different approach
        "reject": 0.20           # Very poor → abandon attempt
    }
    
    def assess(self, question_with_score) -> QualityDecision:
        score = question_with_score.quality_score
        
        if score >= self.thresholds["auto_approve"]:
            return QualityDecision(
                action="approve",
                confidence=score,
                reasoning=f"High quality score {score:.2f} exceeds approval threshold"
            )
        # ... additional decision logic with clear reasoning
```

### Database & Configuration Standards

#### Async Database Operations
```python
# src/derivativ/database/operations.py
async def save_generation_session(session: GenerationSession) -> UUID:
    async with get_db_session() as db:
        try:
            # Save main session record
            db_session = GenerationSessionDB(**session.dict())
            db.add(db_session)
            
            # Save all questions in batch
            for question in session.questions:
                db_question = QuestionDB(**question.dict())
                db.add(db_question)
            
            await db.commit()
            return db_session.id
            
        except Exception as e:
            await db.rollback()
            raise DatabaseError(f"Failed to save session: {e}")
```

#### Environment Configuration
```python
# src/derivativ/core/config.py
class Settings(BaseSettings):
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    
    # LLM APIs
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    anthropic_api_key: str = Field(..., env="ANTHROPIC_API_KEY")
    
    # Performance
    generation_timeout: int = 30
    max_concurrent_generations: int = 5
    
    # Demo settings
    enable_demo_mode: bool = Field(False, env="DEMO_MODE")
    cache_demo_responses: bool = Field(True, env="CACHE_RESPONSES")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
```

---

## 🚀 DERIVATIV COMPETITIVE ADVANTAGES & DEMO STRATEGY

### Technical Differentiation
1. **Multi-Agent AI Coordination**: Specialized agents work together with visible reasoning (unique for hackathons)
2. **Real-Time Quality Control**: Automatic assessment and improvement cycles without human intervention
3. **Production Architecture**: Test-driven development with >95% coverage, async support, proper error handling
4. **Cambridge IGCSE Compliance**: Real curriculum validation, not toy educational examples
5. **Transparent Decision-Making**: All agent reasoning visible for educational and demo purposes

### Demo Narrative (5-Minute Structure)

#### Opening Hook (45 seconds)
*"Derivativ deploys a team of AI specialists that work together like human teachers - a QuestionGenerator creates problems, a MarkerAgent develops solutions, a ReviewAgent assesses quality, and a RefinementAgent improves anything below standard. Watch them collaborate in real-time."*

#### Live Technical Demonstration (3 minutes)
- **Multi-Agent Workflow**: Show live generation with agent reasoning visible
- **Quality Control in Action**: Demonstrate automatic refinement when quality is insufficient  
- **Real Curriculum Compliance**: Generate questions using actual Cambridge IGCSE content references
- **Performance at Scale**: Show sub-30-second generation and concurrent handling

#### Architecture & Impact (1 minute)
- **Production-Ready Code**: Highlight test coverage, async design, error resilience
- **Educational Value**: Real teachers can use this immediately for assessment creation
- **Scalability Potential**: Architecture supports thousands of concurrent users

#### Value Proposition (15 seconds)
*"Derivativ isn't just a demo - it's a production-ready AI education platform that could transform how teachers create assessment materials."*

### Judge Engagement Strategy
- **Technical Judges**: Live code walkthrough showing agent coordination patterns and TDD approach
- **Business Judges**: ROI demonstration - time savings for teachers, personalized learning for students
- **Education Judges**: Real Cambridge compliance validation and pedagogical quality assessment

---

## 🔧 DERIVATIV DEVELOPMENT WORKFLOW

### TDD Development Cycle with Modal + smolagents
```bash
# 1. Write failing tests first
pytest tests/test_modal_agents.py::test_question_generator_modal_function -v
# EXPECTED: Test fails (Red)

# 2. Deploy minimal Modal function to pass tests
modal deploy src/derivativ/agents/question_generator.py
# Implement smolagents-based QuestionGeneratorAgent
# EXPECTED: Test passes (Green)

# 3. Refactor for performance and cost optimization
# Optimize Modal resource allocation, smolagents workflows
# EXPECTED: All tests pass, performance improves (Refactor)
```

### Git Workflow & Standards
```bash
# Branch naming for Derivativ
feature/modal-smolagents-integration
feature/cloudflare-workers-api-gateway
feature/nextjs-ai-sdk-streaming
feature/neon-db-schema-design
fix/modal-cold-start-optimization
test/smolagents-workflow-coverage

# Commit message format
feat: implement Modal + smolagents multi-agent coordination
feat: deploy Cloudflare Workers API gateway with R2 caching
feat: integrate AI SDK streaming components for real-time reasoning
fix: optimize Modal function warm-up for consistent performance
test: add comprehensive smolagents workflow test coverage
```

### AI Agent Collaboration Protocol

#### Before Starting Development
1. **Read current CLAUDE.md**: Understand Modal + smolagents + Cloudflare architecture
2. **Check TodoWrite status**: Identify in-progress vs pending items
3. **Test Modal connection**: Ensure Modal CLI and deployment pipeline works
4. **Verify Neon DB access**: Test database connection and schema setup

#### During Development (TDD with Modal)
1. **Write tests first**: Create failing tests for Modal functions and smolagents workflows
2. **Deploy to Modal iteratively**: Use `modal deploy` for rapid testing cycles
3. **Test smolagents coordination**: Verify agent communication and reasoning
4. **Monitor Cloudflare edge performance**: Test Workers and R2 integration

#### Before Completing Work
1. **Full Modal deployment test**: All agents deployed and communicating via smolagents
2. **Cloudflare Workers validation**: API gateway routes working correctly
3. **End-to-end integration**: Next.js → Cloudflare → Modal → Neon DB flow
4. **Performance benchmarking**: Sub-30-second generation with edge optimization

### Quality Standards for Production Stack
- **Modal Function Performance**: Cold start < 2s, warm execution < 30s total
- **smolagents Coordination**: Agent reasoning clearly logged and trackable
- **Cloudflare Edge Optimization**: Global latency < 100ms for API calls
- **Neon DB Efficiency**: Query performance optimized for agent workflows
- **AI SDK Integration**: Streaming components update smoothly in real-time

---

## 📊 DERIVATIV SUCCESS METRICS & EVALUATION

### Hackathon Success Indicators
1. **Multi-Agent Demo Impact**: Judges clearly see and understand agent coordination
2. **Technical Architecture Recognition**: Production-ready code quality evident  
3. **Educational Value Demonstration**: Real Cambridge IGCSE compliance validation
4. **Performance Under Pressure**: Sub-30-second generation consistently during live demo
5. **Real-World Applicability**: Teachers can immediately use for assessment creation

### Technical Quality Metrics
- **Test Coverage**: >95% on core business logic (`src/derivativ/core/`, `src/derivativ/agents/`)
- **Generation Success Rate**: >90% of requests produce mathematically valid questions
- **Quality Score Distribution**: Average ReviewAgent scores >0.70 for accepted questions
- **Performance Consistency**: 100% of generations complete within 30-second timeout
- **Error Recovery**: Graceful fallback to cached examples in <5 seconds

### Educational Impact Metrics
- **Cambridge Compliance**: 100% of generated questions use valid content references
- **Mathematical Accuracy**: >98% of solutions are mathematically correct
- **Grade Appropriateness**: Difficulty assessment within ±1 grade level 95% of time
- **Question Variety**: No duplicate questions in 1000+ generation sample

### Demo Reliability Metrics
- **Live Demo Success**: 100% uptime during 5-minute presentation
- **Agent Reasoning Visibility**: All decision-making steps clearly displayed in UI
- **WebSocket Stability**: Real-time updates work consistently without disconnects
- **Fallback Strategy Testing**: Backup systems activated successfully when needed

---

## 🏁 DERIVATIV PROJECT SUMMARY

### Project Status: Ready for TDD Implementation
Derivativ is strategically planned as a fresh, production-ready AI education platform with comprehensive technical documentation, clear architecture decisions, and detailed 5-day implementation roadmap.

### Core Innovation: Multi-Agent AI Coordination
The key differentiator is sophisticated agent collaboration with transparent reasoning - QuestionGenerator, MarkerAgent, ReviewAgent, and RefinementAgent working together with visible decision-making that judges can appreciate in real-time.

### Implementation Strategy: Test-Driven Excellence
- **Day 1**: Foundation (tests, models, LLM interface, base agents)
- **Day 2**: AI agent coordination (core value proposition)
- **Day 3**: API & frontend (demo interfaces)
- **Day 4**: Cambridge compliance & performance optimization
- **Day 5**: Integration testing & hackathon presentation prep

### Competitive Advantage for Judges
1. **Production Architecture**: TDD approach with >95% test coverage, async design
2. **Real Educational Value**: Cambridge IGCSE compliance, not toy examples
3. **Transparent AI Reasoning**: Agent decision-making visible for educational purposes
4. **Immediate Usability**: Teachers can use this for real assessment creation

### Success Criteria
- **Technical**: Multi-agent coordination working with visible reasoning
- **Performance**: Sub-30-second generation consistently during live demo  
- **Educational**: Cambridge curriculum compliance and mathematical accuracy
- **Demo Impact**: Judges understand the production-ready sophistication

### Key Message for Judges
*"Derivativ demonstrates the future of AI education - where specialized AI agents collaborate like human teachers to create curriculum-compliant content with transparent decision-making. This isn't a prototype; it's a production-ready platform built with test-driven development that teachers could start using immediately."*

---

**Implementation Ready**: All planning complete. Time to start building with **TODO 1.1: Initialize Derivativ Project Structure** using the comprehensive TDD approach outlined in `tdd_todo.md` and `STRATEGIC_PLAN.md`.