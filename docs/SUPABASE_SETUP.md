# Supabase + FastAPI Setup Guide

This guide walks through setting up Supabase database integration with the Derivativ FastAPI backend.

## 🚀 Quick Start

1. **Get Supabase credentials:**
   ```bash
   python tools/setup_api_keys.py
   # Follow prompts to add SUPABASE_URL and SUPABASE_ANON_KEY
   ```

2. **Run database migrations:**
   ```bash
   # Apply migrations in your Supabase dashboard SQL editor
   # Or use Supabase CLI if available
   supabase migration up
   ```

3. **Start FastAPI with database:**
   ```bash
   uvicorn src.api.main:app --reload
   ```

4. **Test integration:**
   ```bash
   python -m pytest tests/integration/ -m integration
   ```

## 📋 Database Schema

### Core Tables

#### `questions` Table
Stores generated questions with hybrid storage pattern:
- **Flattened fields**: Fast querying (tier, marks, command_word, etc.)
- **JSONB field**: Complete Pydantic model for data fidelity

#### `generation_sessions` Table
Tracks multi-agent question generation workflows:
- Session metadata (topic, tier, status)
- Complete agent results and reasoning
- Performance metrics and timing

#### Enum Tables
Reference tables for data consistency:
- `tiers` - Core/Extended levels
- `command_words` - Cambridge assessment terms
- `calculator_policies` - Calculator usage rules
- `generation_statuses` - Workflow states
- `question_origins` - Source types

## 🔄 Real-time Features

### Supabase Realtime Integration

The system includes live database updates via Supabase Realtime:

```python
# WebSocket endpoint for live generation updates
@router.websocket("/ws/generate/{session_id}")
async def websocket_generate(websocket: WebSocket, session_id: str):
    # Streams both:
    # 1. Agent generation updates
    # 2. Database changes in real-time
```

### Real-time Capabilities
- **Live agent progress**: See reasoning steps as they happen
- **Database sync**: Automatic UI updates when questions are saved
- **Status changes**: Track generation workflow progress
- **Performance monitoring**: Live processing time updates

## 🧪 Testing Levels

### Unit Tests (Fast, Isolated)
```bash
python -m pytest tests/unit/ -v
```
- Mock Supabase client
- Test business logic
- Repository pattern validation

### Integration Tests (Real Database)
```bash
# Requires SUPABASE_URL and SUPABASE_ANON_KEY
python -m pytest tests/integration/ -m integration -v
```
- Actual database operations
- Schema validation
- End-to-end repository tests

### E2E Tests (Full System)
```bash
python -m pytest tests/e2e/ -m e2e -v
```
- FastAPI + Supabase + Realtime
- Complete user workflows
- Performance validation

## 📊 Architecture Benefits

### Hybrid Storage Pattern
- **Query Performance**: Indexed flattened fields for fast filtering
- **Data Fidelity**: Complete JSONB for full model preservation
- **Schema Evolution**: Easy to add new fields without migration

### Real-time Streaming
- **Agent Transparency**: Live reasoning display for educational value
- **Database Sync**: Automatic UI updates without polling
- **Scalable**: Edge-optimized WebSocket connections

### Production Ready
- **Row Level Security**: Fine-grained access control
- **Automatic Scaling**: Supabase handles connection pooling
- **Backup & Recovery**: Built-in database management
- **Global Distribution**: Edge database locations

## 🔧 Development Workflow

### Local Development
1. Set up environment variables in `.env`
2. Run FastAPI server with auto-reload
3. Use integration tests for database validation
4. Mock external services for unit tests

### Database Changes
1. Create migration in `supabase/migrations/`
2. Apply via Supabase dashboard or CLI
3. Update repository methods if needed
4. Add integration tests for new features

### Performance Monitoring
- Track query performance with built-in indexes
- Monitor real-time connection stability
- Validate generation speed targets (<30s)
- Test concurrent request handling

## 🎯 Next Steps

### Ready for Production
- [x] Database schema with proper indexes
- [x] Real-time streaming infrastructure
- [x] Comprehensive test coverage
- [x] Repository pattern for clean separation

### Frontend Integration
- WebSocket client for live updates
- Question display components
- Generation progress indicators
- Error handling and fallbacks

### Scaling Considerations
- Connection pooling optimization
- Read replicas for query performance
- Caching layer for frequently accessed data
- Load testing for concurrent users

## 🐛 Troubleshooting

### Common Issues

**Import Errors**:
```bash
pip install realtime-py>=0.3.0
```

**Connection Failures**:
- Verify SUPABASE_URL format (https://xxx.supabase.co)
- Check SUPABASE_ANON_KEY is correct
- Ensure Row Level Security policies allow access

**Test Failures**:
- Run `python tools/setup_api_keys.py` to configure
- Check environment variables are loaded
- Verify database migrations are applied

### Performance Issues
- Check query indexes are being used
- Monitor real-time connection count
- Validate generation times with caching
- Test with multiple concurrent sessions
