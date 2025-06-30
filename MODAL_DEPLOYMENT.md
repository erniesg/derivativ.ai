# 🚀 Deploy Derivativ AI to Modal

## Quick Start (3 Steps)

```bash
# 1. Install and authenticate Modal
pip install modal
modal setup

# 2. Set up secrets (optional - works in demo mode without API keys)
python setup_modal_secrets.py

# 3. Deploy your FastAPI app  
modal deploy deploy.py
```

**That's it!** Modal gives you a URL like: `https://derivativ-ai--username.modal.run`

### Test Your Deployment
```bash
# Test locally first
python deploy.py

# Test the live API
curl https://your-url.modal.run/health
```

## 📡 Using Your Deployed API

### Web Interface
After deployment, Modal provides you with a public URL. You can access:
- **API Documentation**: `https://your-app-url.modal.run/docs`
- **Health Check**: `https://your-app-url.modal.run/health`

### API Endpoints

#### Generate Questions
```bash
curl -X POST "https://your-app-url.modal.run/api/questions/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Algebra",
    "grade_level": 9,
    "num_questions": 5,
    "difficulty": "medium"
  }'
```

#### Generate Document
```bash
curl -X POST "https://your-app-url.modal.run/api/documents/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "material_type": "worksheet",
    "topics": ["Algebra", "Linear Equations"],
    "detail_level": 5,
    "grade_level": 9
  }'
```

### Python Client Example
```python
import requests

# Your Modal app URL
BASE_URL = "https://your-app-url.modal.run"

# Generate questions
response = requests.post(f"{BASE_URL}/api/questions/generate", json={
    "topic": "Algebra",
    "grade_level": 9,
    "num_questions": 3,
    "difficulty": "medium"
})

questions = response.json()
print(f"Generated {len(questions['questions'])} questions")
```

### JavaScript/Frontend Integration
```javascript
const baseUrl = 'https://your-app-url.modal.run';

async function generateQuestions(topic, gradeLevel = 9) {
  const response = await fetch(`${baseUrl}/api/questions/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      topic: topic,
      grade_level: gradeLevel,
      num_questions: 5,
      difficulty: 'medium'
    })
  });
  
  return await response.json();
}

// Usage
generateQuestions('Algebra')
  .then(result => console.log(result))
  .catch(error => console.error(error));
```

## 🔧 Modal Functions

Your deployment includes these individual Modal functions:

### 1. FastAPI Web App
- **Function**: `fastapi_app`
- **URL**: Provided by Modal after deployment
- **Description**: Full FastAPI application with all endpoints

### 2. Standalone Question Generation
```python
import modal

# Call the function directly
questions = modal.Function.lookup("derivativ-ai", "generate_questions").remote(
    topic="Algebra",
    grade_level=9,
    num_questions=5,
    difficulty="medium"
)
```

### 3. Document Generation
```python
import modal

# Generate educational documents
document = modal.Function.lookup("derivativ-ai", "generate_document").remote(
    material_type="worksheet",
    topics=["Algebra", "Linear Equations"],
    detail_level=5,
    grade_level=9
)
```

## 🔐 Environment Variables

The deployment supports these environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | No* | OpenAI API key for question generation |
| `ANTHROPIC_API_KEY` | No* | Anthropic API key for AI agents |
| `GOOGLE_API_KEY` | No* | Google Gemini API key |
| `SUPABASE_URL` | No* | Supabase database URL |
| `SUPABASE_KEY` | No* | Supabase API key |
| `DEMO_MODE` | No | Set to "true" to enable demo mode (default if API keys missing) |

*Required for full functionality, but demo mode works without them.

## 🐛 Troubleshooting

### Common Issues

1. **Deployment fails with dependency errors**:
   ```bash
   # Clear Modal cache and redeploy
   modal volume delete cache-vol
   modal deploy deploy.py
   ```

2. **Secret not found error**:
   ```bash
   # Recreate secrets
   python setup_modal_secrets.py
   ```

3. **Import errors in deployment**:
   - Check that all files are copied correctly in `deploy.py`
   - Verify Python path is set correctly

### Debug Mode
To enable detailed logging:
```bash
modal deploy deploy.py --stream-logs
```

### View Logs
```bash
modal logs derivativ-ai
```

## 📊 Monitoring

### View App Status
```bash
modal app list
```

### Function Logs
```bash
modal logs derivativ-ai::fastapi_app
```

### Performance Metrics
Access the Modal dashboard at https://modal.com/apps to view:
- Request latency
- Error rates
- Resource usage
- Cost breakdown

## 💰 Cost Optimization

### Current Configuration
- **CPU**: 2 cores for AI workloads
- **Memory**: 2GB for main app, 1GB for functions
- **Keep Warm**: 2 containers to reduce cold starts
- **Timeout**: 300s for complex AI operations

### Cost-Saving Tips
1. **Reduce keep_warm** for lower traffic:
   ```python
   keep_warm=0  # Allow cold starts to save costs
   ```

2. **Optimize timeouts** based on actual usage:
   ```python
   timeout=120  # Reduce if operations complete faster
   ```

3. **Use demo mode** for development:
   ```python
   secrets=[modal.Secret.from_dict({"DEMO_MODE": "true"})]
   ```

## 🔄 Updates and Redeployment

### Update Application
```bash
# Make your changes, then redeploy
modal deploy deploy.py
```

### Rolling Updates
Modal automatically handles rolling updates with zero downtime.

### Rollback
If needed, you can deploy a previous version:
```bash
git checkout previous-commit
modal deploy deploy.py
```

## 🎯 Production Checklist

- [ ] Environment variables set in Modal secrets
- [ ] API keys tested and working
- [ ] Deployment successful with no errors
- [ ] Health check endpoint responding
- [ ] API documentation accessible
- [ ] Frontend integration tested
- [ ] Monitoring and alerts configured
- [ ] Cost limits set in Modal dashboard

## 🤝 Support

For issues with:
- **Modal platform**: Check [Modal documentation](https://modal.com/docs)
- **Derivativ AI**: Check application logs or create an issue
- **API integration**: Refer to the `/docs` endpoint on your deployed app