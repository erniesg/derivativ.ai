# Multi-Tab Question Generation Commands

Run these commands in **separate terminal tabs** to populate your database quickly with diverse questions.

## **🚀 Quick Start (Open 6 tabs)**

### **Tab 1: Basic Arithmetic (Grades 1-3)**
```bash
python main.py --debug generate --grades 1 2 3 --count 5 --calculator-policy not_allowed
```

### **Tab 2: Middle School Math (Grades 4-6)**
```bash
python main.py --debug generate --grades 4 5 6 --count 4 --calculator-policy allowed
```

### **Tab 3: Advanced Math (Grades 7-9)**
```bash
python main.py --debug generate --grades 7 8 9 --count 3 --calculator-policy varies_by_question
```

### **Tab 4: Algebra Focus**
```bash
python main.py --debug generate --grades 5 6 7 --subject-refs C2.2 C2.5 C2.6 --count 4
```

### **Tab 5: Geometry Focus**
```bash
python main.py --debug generate --grades 6 7 8 --subject-refs C4.1 C4.2 C4.3 C4.5 --count 3
```

### **Tab 6: Mixed Topics**
```bash
python main.py --debug generate --grades 4 5 6 7 8 --count 2 --model gpt-4o-mini
```

---

## **📊 Expected Results**
- **Tab 1**: 15 questions (5 × 3 grades) - Basic arithmetic
- **Tab 2**: 12 questions (4 × 3 grades) - Middle school
- **Tab 3**: 9 questions (3 × 3 grades) - Advanced
- **Tab 4**: 12 questions (4 × 3 grades) - Algebra focus
- **Tab 5**: 9 questions (3 × 3 grades) - Geometry focus
- **Tab 6**: 10 questions (2 × 5 grades) - Mixed topics

**Total: ~67 questions** in parallel generation

---

## **⚡ Speed Optimization Commands**

### **Fast Generation (GPT-4o-mini only)**
```bash
# Tab A
python main.py generate --grades 1 2 3 4 5 --count 3 --model gpt-4o-mini

# Tab B
python main.py generate --grades 6 7 8 9 --count 3 --model gpt-4o-mini
```

### **High Quality (GPT-4o only)**
```bash
# Tab C
python main.py generate --grades 5 6 7 --count 2 --model gpt-4o --calculator-policy allowed

# Tab D
python main.py generate --grades 7 8 9 --count 2 --model gpt-4o --calculator-policy varies_by_question
```

---

## **🎯 Subject-Specific Batches**

### **Arithmetic & Number Work**
```bash
python main.py generate --grades 2 3 4 --subject-refs C1.1 C1.2 C1.4 C1.6 --count 3
```

### **Algebra**
```bash
python main.py generate --grades 5 6 7 8 --subject-refs C2.1 C2.2 C2.5 C2.6 C2.10 --count 2
```

### **Geometry**
```bash
python main.py generate --grades 6 7 8 9 --subject-refs C4.1 C4.2 C4.3 C4.5 C4.6 --count 2
```

### **Statistics & Probability**
```bash
python main.py generate --grades 7 8 9 --subject-refs C7.1 C8.3 C9.3 C9.5 --count 2
```

---

## **🔍 Monitoring Commands**

### **Check Progress**
```bash
python main.py list --limit 20
```

### **Grade Distribution**
```bash
python main.py stats
```

### **Filter by Grade**
```bash
python main.py list --grade 5 --limit 10
```

---

## **📈 Production Workflow Commands**

### **Small Batches for Testing**
```bash
# Tab 1: Test basic functionality
python main.py --debug generate --grades 5 --count 2

# Tab 2: Test different models
python main.py --debug generate --grades 6 --count 2 --model gpt-4o

# Tab 3: Test subject references
python main.py --debug generate --grades 7 --subject-refs C2.5 C4.2 --count 2
```

### **Large Batches for Production**
```bash
# Tab 1: Comprehensive generation
python main.py generate --grades 1 2 3 4 5 6 7 8 9 --count 10

# Tab 2: Subject-focused generation
python main.py generate --grades 5 6 7 8 --subject-refs C1.6 C2.2 C4.1 C7.1 --count 5

# Tab 3: Calculator policy variations
python main.py generate --grades 6 7 8 --calculator-policy varies_by_question --count 5
```

---

## **⚠️ Important Notes**

1. **Database Persistence**: All questions are automatically saved to your NEON database
2. **No Conflicts**: Each generation creates unique question IDs
3. **Debug Mode**: Use `--debug` to see detailed generation process
4. **Rate Limits**: OpenAI has rate limits, so multiple tabs help distribute load
5. **Monitoring**: Check `python main.py stats` regularly to track progress

---

## **🎉 Quick Population Strategy**

**Goal: 100 questions in ~10 minutes**

```bash
# Open 4 tabs simultaneously:

# Tab 1 (25 questions):
python main.py generate --grades 1 2 3 4 5 --count 5 --model gpt-4o-mini

# Tab 2 (25 questions):
python main.py generate --grades 6 7 8 9 --count 6 --model gpt-4o-mini

# Tab 3 (25 questions):
python main.py generate --grades 4 5 6 7 --count 6 --model gpt-4o

# Tab 4 (25 questions):
python main.py generate --grades 5 6 7 8 9 --count 5 --model gpt-4o
```

**Result: ~100 questions across all grades with mixed models**
