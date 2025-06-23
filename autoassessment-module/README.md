# Quiz Backend System

Adaptive learning quiz backend with **Exponential Weighted Moving Average (EWMA)** performance tracking and **dynamic difficulty adjustment**.

## 🎯 Features

- **Exponential Weighted Moving Average**: Tracks student performance with exponential decay
- **Dynamic Difficulty**: Softmax-based question distribution adjustment
- **Grade Padding**: Conservative starting scores to encourage growth
- **6 Math Topics**: Number, Geometry, Trig, Coordinate Geometry, Transformation, Probability
- **1-9 Scale**: Precise performance grading per topic

## 🏗️ Architecture

### Database Tables
```
topics                    → Mathematical topics (6 topics)
questions                 → Question bank with difficulty levels
quiz_sessions            → Individual quiz attempts
student_topic_performance → EWMA scores per topic
quiz_responses           → Question answers
quiz_results             → Quiz performance summaries
wma_history              → Performance change history
```

### Services
- **Quiz Service**: EWMA calculations, difficulty adjustment, question selection
- **Auth Controller**: User profiles and performance initialization
- **Functional Design**: Pure JavaScript functions, no classes

## 📊 EWMA Algorithm

### Exponential Weighted Moving Average
```javascript
EWMA = (1 - α) × previous_EWMA + α × new_score
```

**Parameters:**
- `α = 0.3` (smoothing factor)
- `new_score` = Current quiz performance (1-9)
- `previous_EWMA` = Previous exponential weighted moving average

### Grade Padding
```javascript
const PADDING_FACTOR = 0.8;
const startingGrade = 4.0 * PADDING_FACTOR; // = 3.2
```

Students start at **3.2/9** to encourage genuine improvement.

### Dynamic Difficulty (Softmax)

Based on average EWMA across topics:

| Performance | EWMA Range | Easy | Medium | Hard |
|------------|------------|------|--------|------|
| Low        | < 3.4      | 68%  | 24%    | 8%   |
| Medium     | 3.4-6.6    | 34%  | 44%    | 22%  |
| High       | > 6.6      | 12%  | 39%    | 49%  |

## 🎓 How It Works

1. **Student Profile Created** → EWMA initialized at 3.2/9 for all topics
2. **Quiz Started** → Questions selected based on current EWMA distribution
3. **Answers Submitted** → Individual responses tracked
4. **Quiz Completed** → EWMA updated per topic using exponential smoothing
5. **Next Quiz** → Difficulty automatically adjusted based on new EWMA scores

The exponential weighting gives more importance to recent performance while still considering historical data, making the system responsive but not overly reactive to single quiz results.

## 🎓 Educational Benefits

### Growth Mindset
- Students start below average, must earn progress
- Every improvement feels genuinely rewarding
- Prevents false confidence from high initial scores

### Adaptive Learning
- Questions automatically match student ability
- Struggling students get more support (easier questions)
- Advanced students stay challenged (harder questions)

### Data-Driven Insights
- Track improvement over time with EWMA
- Identify strong/weak topics per student
- Measure learning velocity and trends

## 🛡️ Security

- **Row Level Security**: Database policies restrict data access
- **JWT Authentication**: Secure token-based auth with Supabase
- **Input Validation**: Joi schema validation on all endpoints
- **Rate Limiting**: Protection against API abuse
- **CORS Configuration**: Controlled cross-origin access
