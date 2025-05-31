# No-Diagram Question Generation Commands
*Cambridge IGCSE Mathematics questions that don't require visual diagrams*

These commands focus on topics that can be expressed purely through text:
- Number operations and calculations
- Algebraic expressions and equations
- Area/volume calculations with given dimensions
- Statistics and probability with data tables
- Coordinate geometry with given coordinates

## 🚀 Basic No-Diagram Commands (By Grade Level)

### **Grade 1 (No Diagrams)**
```bash
python main.py --debug generate --grades 1 --subject-refs E1.2 E9.5 E1.5 E2.9 E1.9 E1.7 --count 4
```

### **Grade 2 (No Diagrams)**
```bash
python main.py --debug generate --grades 2 --subject-refs E8.2 E8.1 E9.3 E3.1 C3.4 C1.8 --count 4
```

### **Grade 3 (No Diagrams)**
```bash
python main.py --debug generate --grades 3 --subject-refs C7.3 C1.9 C5.2 E1.5 E2.5 E2.2 --count 4
```

### **Grade 4 (No Diagrams)**
```bash
python main.py --debug generate --grades 4 --subject-refs C3.1 C2.5 C3.5 C3.7 C9.1 E2.7 --count 4
```

### **Grade 5 (No Diagrams)**
```bash
python main.py --debug generate --grades 5 --subject-refs C3.7 C3.8 E2.7 C2.10 E2.5 C1.10 --count 4
```

### **Grade 6 (No Diagrams)**
```bash
python main.py --debug generate --grades 6 --subject-refs E2.10 E7.5 C7.5 C1.9 E3.5 E3.2 --count 4
```

### **Grade 7 (No Diagrams)**
```bash
python main.py --debug generate --grades 7 --subject-refs C9.5 C7.4 E7.3 C1.7 C9.1 E1.6 --count 4
```

### **Grade 8 (No Diagrams)**
```bash
python main.py --debug generate --grades 8 --subject-refs C7.7 E3.3 E2.6 E2.3 C1.9 C8.1 --count 4
```

### **Grade 9 (No Diagrams)**
```bash
python main.py --debug generate --grades 9 --subject-refs C7.4 C8.4 E7.4 C9.3 C1.3 C5.4 --count 4
```

## 📊 Topic-Specific No-Diagram Commands

### **Number Operations**
```bash
python main.py generate --grades 3 4 5 6 --subject-refs C1.1 C1.2 C1.4 C1.6 C1.8 --count 3
```

### **Algebra**
```bash
python main.py generate --grades 3 4 5 6 --subject-refs C2.1 C2.2 C2.5 C2.6 C2.9 --count 3
```

### **Percentage & Ratio**
```bash
python main.py generate --grades 3 4 5 6 --subject-refs C1.7 C3.1 C3.2 C3.3 --count 3
```

### **Text-Based Geometry**
```bash
python main.py generate --grades 3 4 5 6 --subject-refs C5.2 C5.4 --count 3
```

### **Statistics**
```bash
python main.py generate --grades 3 4 5 6 --subject-refs C7.1 C7.2 C7.5 C8.3 --count 3
```

### **Probability**
```bash
python main.py generate --grades 3 4 5 6 --subject-refs C9.1 C9.2 C9.3 --count 3
```

### **Extended Number**
```bash
python main.py generate --grades 5 6 7 8 --subject-refs E1.1 E1.3 E1.5 E2.1 --count 3
```

### **Extended Algebra**
```bash
python main.py generate --grades 5 6 7 8 --subject-refs E2.5 E2.7 E2.9 E3.1 --count 3
```

### **Extended Statistics**
```bash
python main.py generate --grades 5 6 7 8 --subject-refs E7.1 E7.3 E8.1 E9.1 --count 3
```

## 🎲 Random No-Diagram Sampling Commands

### **Random Foundation (Grades 1-6)**
```bash
python main.py generate --grades 1 2 3 4 5 6 --subject-refs C2.7 C3.5 C3.2 C2.3 C2.8 C1.2 C1.4 C1.9 C1.6 C1.3 --count 2
```

### **Random Higher (Grades 6-9)**
```bash
python main.py generate --grades 6 7 8 9 --subject-refs C2.10 C9.4 E9.2 C3.1 C2.6 C1.6 E9.5 C2.2 C1.8 E9.3 C2.5 C8.4 --count 2
```

### **Random Mixed (All Topics)**
```bash
python main.py generate --grades 4 5 6 7 8 --subject-refs C1.10 E3.4 C3.1 E7.1 E8.3 C1.4 C2.7 C5.4 --count 3
```

## 📐 Text-Based Geometry Commands
*(Geometry questions using given dimensions, no visual diagrams)*

### **Area & Perimeter Calculations**
```bash
python main.py generate --grades 4 5 6 --subject-refs C5.2 --count 4
```

### **Volume & Surface Area**
```bash
python main.py generate --grades 6 7 8 --subject-refs C5.4 E5.4 --count 3
```

### **Combined Mensuration**
```bash
python main.py generate --grades 5 6 7 8 --subject-refs C5.2 C5.4 E5.2 E5.4 --count 3
```

## ⚡ Batch No-Diagram Population Commands

### **Quick 50 Questions (5 Tabs)**

**Tab 1:**
```bash
python main.py generate --grades 1 2 3 --subject-refs E8.1 C8.5 C7.7 E9.6 E2.2 E3.7 E2.4 C2.3 --count 10
```

**Tab 2:**
```bash
python main.py generate --grades 3 4 5 --subject-refs E8.5 C3.6 C1.5 E1.2 C8.5 E3.8 C9.2 E8.3 --count 10
```

**Tab 3:**
```bash
python main.py generate --grades 5 6 7 --subject-refs E7.1 C7.3 C7.1 E7.6 C9.1 C8.5 E3.2 E7.7 --count 10
```

**Tab 4:**
```bash
python main.py generate --grades 6 7 8 --subject-refs C1.2 E1.7 C3.8 C8.5 C9.5 C9.1 C2.7 C8.2 --count 10
```

**Tab 5:**
```bash
python main.py generate --grades 7 8 9 --subject-refs E3.8 C3.1 E1.1 E7.5 E7.1 E9.6 C1.2 E2.2 --count 10
```

## 📋 Notes
- All these topics work well with `raw_text_content` for frontend parsing
- Questions include dimensions/data in text descriptions
- No visual diagram generation required
- Perfect for automated generation and testing
- Frontend can render these as pure text or simple formatted text
