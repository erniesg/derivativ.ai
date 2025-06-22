# Live Testing Guide - Derivativ AI Platform

This guide provides step-by-step instructions for testing the complete Derivativ system with real APIs and database.

## 🚀 Quick Start (5 minutes)

### 1. **Environment Setup**
```bash
# Clone and navigate to project
git clone <repo-url>
cd derivativ.ai

# Install dependencies
pip install -r requirements.txt

# Set up API keys and database
python tools/setup_api_keys.py
```

### 2. **Test System Health**
```bash
# Check all services are configured
python -c "
from src.api.dependencies import get_system_health
import json
print(json.dumps(get_system_health(), indent=2))
"
```

### 3. **Start the API Server**
```bash
# Start FastAPI server with auto-reload
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. **Test API Endpoints**
```bash
# Health check
curl http://localhost:8000/health

# Generate a question (requires Supabase + LLM API keys)
curl -X POST http://localhost:8000/api/questions/generate \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "algebra",
    "tier": "Core",
    "marks": 3,
    "command_word": "Calculate"
  }'
```

---

## 📋 Comprehensive Testing Checklist

### **Environment Prerequisites**

- [ ] **Python 3.10+** installed
- [ ] **Git** repository cloned and up to date
- [ ] **Dependencies** installed via `pip install -r requirements.txt`

### **API Keys Required**

#### LLM Providers (at least one required)
- [ ] **OpenAI API Key** - Get from [platform.openai.com](https://platform.openai.com/api-keys)
- [ ] **Anthropic API Key** - Get from [console.anthropic.com](https://console.anthropic.com/)
- [ ] **Google AI API Key** - Get from [makersuite.google.com](https://makersuite.google.com/app/apikey)

#### Database (required for persistence)
- [ ] **Supabase Project URL** - Create project at [supabase.com](https://supabase.com)
- [ ] **Supabase Anon Key** - Get from project settings → API

#### Optional (for enhanced features)
- [ ] **Hugging Face Token** - Get from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### **Database Setup**

#### Create Supabase Project
1. Go to [supabase.com](https://supabase.com) and create new project
2. Wait for project to initialize (~2 minutes)
3. Go to **Settings → API** and copy:
   - Project URL (e.g., `https://abc123.supabase.co`)
   - Anon public key (starts with `eyJ...`)

#### Apply Database Migrations
1. **Via Supabase Dashboard** (Recommended):
   - Go to **SQL Editor** in Supabase dashboard
   - Copy and run each migration file in order:
     - `supabase/migrations/001_create_questions_table.sql`
     - `supabase/migrations/002_create_generation_sessions_table.sql`
     - `supabase/migrations/003_create_enum_tables.sql`

2. **Via Supabase CLI** (Alternative):
   ```bash
   # Install Supabase CLI
   npm install -g supabase

   # Link to your project
   supabase link --project-ref YOUR_PROJECT_REF

   # Run migrations
   supabase db push
   ```

---

## 🧪 Testing Scenarios

### **Scenario 1: Basic API Health Check**

```bash
# Test basic connectivity
curl -X GET http://localhost:8000/health

# Expected response (healthy):
{
  "status": "healthy",
  "service": "derivativ-api",
  "database": "healthy",
  "supabase": "connected",
  "realtime": "configured"
}

# Expected response (configuration issues):
{
  "status": "degraded",
  "service": "derivativ-api",
  "database": "unhealthy",
  "error": "Supabase not configured..."
}
```

### **Scenario 2: Question Generation (Full Workflow)**

```bash
# Test question generation
curl -X POST http://localhost:8000/api/questions/generate \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "algebra",
    "tier": "Core",
    "grade_level": 9,
    "marks": 4,
    "command_word": "Solve",
    "calculator_policy": "not_allowed"
  }'

# Expected response structure:
{
  "session_id": "uuid-here",
  "questions": [
    {
      "question_id_global": "uuid-here",
      "marks": 4,
      "command_word": "Solve",
      "raw_text_content": "Solve the equation...",
      "solution_and_marking_scheme": {...},
      "taxonomy": {...}
    }
  ],
  "status": "candidate",
  "agent_results": [...]
}
```

### **Scenario 3: Real-time WebSocket Testing**

```javascript
// Test WebSocket connection (run in browser console)
const ws = new WebSocket('ws://localhost:8000/api/ws/generate/test-session');

ws.onopen = () => {
  console.log('WebSocket connected');

  // Send generation request
  ws.send(JSON.stringify({
    action: 'generate',
    request: {
      topic: 'geometry',
      tier: 'Extended',
      marks: 6,
      command_word: 'Prove'
    }
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);

  // Expected message types:
  // - connection_established
  // - generation_started
  // - agent_update
  // - generation_complete
};
```

### **Scenario 4: Database Integration Testing**

```bash
# List generated questions
curl "http://localhost:8000/api/questions?tier=Core&limit=5"

# Get specific question
curl "http://localhost:8000/api/questions/{question_id_global}"

# List generation sessions
curl "http://localhost:8000/api/sessions?status=candidate&limit=10"
```

### **Scenario 5: Performance Testing**

```bash
# Test multiple concurrent requests
for i in {1..5}; do
  curl -X POST http://localhost:8000/api/questions/generate \
    -H "Content-Type: application/json" \
    -d '{"topic": "statistics", "tier": "Core", "marks": 3}' &
done
wait

# Monitor generation time (should be <30 seconds)
time curl -X POST http://localhost:8000/api/questions/generate \
  -H "Content-Type: application/json" \
  -d '{"topic": "calculus", "tier": "Extended", "marks": 8}'
```

---

## 🔧 Advanced Testing

### **Integration Tests**

```bash
# Run integration tests (requires Supabase setup)
python -m pytest tests/integration/ -v -m integration

# Run E2E tests (full system)
python -m pytest tests/e2e/ -v -m e2e

# Run performance tests
python -m pytest tests/performance/ -v -m performance
```

### **Agent Testing**

```bash
# Test smolagents integration
python examples/smolagents_interactive_demo.py

# Test multi-agent workflow
python examples/smolagents_tools_demo.py

# Test LLM connectivity
python examples/live_apis.py
```

### **Load Testing**

```bash
# Install load testing tool
pip install locust

# Run load test
locust -f tests/performance/locust_load_test.py --host=http://localhost:8000
```

---

## 🐛 Troubleshooting

### **Common Issues**

#### **503 Service Unavailable**
- **Cause**: Missing Supabase credentials
- **Fix**: Run `python tools/setup_api_keys.py` and add SUPABASE_URL + SUPABASE_ANON_KEY

#### **Generation Takes Too Long**
- **Cause**: Missing or slow LLM API keys
- **Fix**: Ensure at least one LLM provider API key is configured

#### **WebSocket Connection Fails**
- **Cause**: Missing Realtime configuration
- **Fix**: Verify Supabase credentials and realtime permissions

#### **Database Connection Errors**
- **Cause**: Missing database schema
- **Fix**: Apply database migrations via Supabase dashboard

### **Debug Commands**

```bash
# Check environment variables
python -c "
import os
print('Environment status:')
for key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'SUPABASE_URL', 'SUPABASE_ANON_KEY']:
    print(f'{key}: {'✅' if os.getenv(key) else '❌'}')
"

# Test database connectivity
python -c "
from src.api.dependencies import check_database_health
import json
print('Database health:', json.dumps(check_database_health(), indent=2))
"

# Verify API dependencies
python -c "
try:
    from src.api.main import app
    print('✅ FastAPI app imports successfully')
except Exception as e:
    print(f'❌ FastAPI import error: {e}')
"
```

### **Performance Benchmarks**

| Metric | Target | Measurement |
|--------|--------|-------------|
| Question Generation | <30 seconds | Time from API request to response |
| Health Check | <100ms | Simple endpoint response time |
| WebSocket Connection | <200ms | Connection establishment time |
| Database Query | <500ms | Question retrieval time |
| Concurrent Requests | 5+ simultaneous | Multiple generation requests |

---

## 🎯 Success Criteria

### **Functional Tests**
- [ ] Health endpoint returns system status
- [ ] Question generation completes successfully
- [ ] Generated questions are mathematically valid
- [ ] Questions are saved to and retrieved from database
- [ ] WebSocket streaming works in real-time
- [ ] Agent reasoning is visible and logical

### **Performance Tests**
- [ ] Generation completes in <30 seconds
- [ ] System handles 5+ concurrent requests
- [ ] Memory usage remains stable
- [ ] No memory leaks during extended operation

### **Integration Tests**
- [ ] All test suites pass (unit, integration, e2e)
- [ ] Supabase database operations work correctly
- [ ] LLM provider switching works seamlessly
- [ ] Error handling is graceful and informative

---

## 📊 Monitoring & Metrics

### **Real-time Monitoring**

```bash
# Monitor API logs
tail -f logs/derivativ-api.log

# Monitor database performance
# (via Supabase dashboard → Database → Performance)

# Monitor system resources
htop  # or Activity Monitor on macOS
```

### **Key Metrics to Track**

1. **Generation Success Rate**: >90% of requests should complete successfully
2. **Average Response Time**: <30 seconds for question generation
3. **Database Performance**: <500ms for queries
4. **Error Rate**: <5% of requests should fail
5. **WebSocket Stability**: Connections should remain stable for >5 minutes

### **Production Readiness Checklist**

- [ ] All tests passing consistently
- [ ] Performance targets met under load
- [ ] Error handling graceful and informative
- [ ] Security considerations addressed (API keys, database access)
- [ ] Monitoring and logging in place
- [ ] Documentation complete and accurate

---

**Ready for Live Demo!** 🚀

The system is now ready for live demonstration with real API calls, database persistence, and real-time streaming capabilities.
