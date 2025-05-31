#!/usr/bin/env python3
"""
Generate Questions Without Diagrams

This script generates Cambridge IGCSE questions from topics that typically
don't require visual diagrams, including:
- Area/perimeter calculations (given dimensions)
- Volume/surface area calculations
- Number operations and algebra
- Statistics and probability calculations
- Coordinate geometry (given coordinates)

All generated questions use text descriptions instead of visual diagrams.
"""

import random
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Non-diagram subject references
NO_DIAGRAM_REFS = [
    # Number and Algebra (always text-based)
    "C1.1", "C1.2", "C1.3", "C1.4", "C1.5", "C1.6", "C1.7", "C1.8", "C1.9", "C1.10",
    "C2.1", "C2.2", "C2.3", "C2.4", "C2.5", "C2.6", "C2.7", "C2.8", "C2.9", "C2.10",
    "C3.1", "C3.2", "C3.3", "C3.4", "C3.5", "C3.6", "C3.7", "C3.8",

    # Mensuration (when dimensions are given in text)
    "C5.2",  # Area and perimeter (text descriptions)
    "C5.4",  # Surface area and volume (text descriptions)

    # Statistics and Probability (text/data based)
    "C7.1", "C7.2", "C7.3", "C7.4", "C7.5", "C7.6", "C7.7",
    "C8.1", "C8.2", "C8.3", "C8.4", "C8.5",
    "C9.1", "C9.2", "C9.3", "C9.4", "C9.5",

    # Extended references for higher grades
    "E1.1", "E1.2", "E1.3", "E1.4", "E1.5", "E1.6", "E1.7", "E1.8", "E1.9",
    "E2.1", "E2.2", "E2.3", "E2.4", "E2.5", "E2.6", "E2.7", "E2.8", "E2.9", "E2.10",
    "E3.1", "E3.2", "E3.3", "E3.4", "E3.5", "E3.6", "E3.7", "E3.8",
    "E5.2", "E5.4",  # Extended mensuration
    "E7.1", "E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7",
    "E8.1", "E8.2", "E8.3", "E8.4", "E8.5",
    "E9.1", "E9.2", "E9.3", "E9.4", "E9.5", "E9.6"
]

# Geometry topics that work well without diagrams (when dimensions are provided)
TEXT_GEOMETRY_REFS = [
    "C5.2",  # Area and perimeter - rectangles, circles with given dimensions
    "C5.4",  # Volume - cubes, cylinders with given dimensions
    "E5.2",  # Extended area calculations
    "E5.4",  # Extended volume calculations
]

def generate_basic_commands():
    """Generate basic no-diagram commands for each grade level"""
    print("## 🚀 Basic No-Diagram Commands (By Grade Level)")
    print()

    for grade in range(1, 10):
        # Sample random references for this grade
        sample_refs = random.sample(NO_DIAGRAM_REFS, min(6, len(NO_DIAGRAM_REFS)))
        refs_str = " ".join(sample_refs)

        print(f"### **Grade {grade} (No Diagrams)**")
        print(f"```bash")
        print(f"python main.py --debug generate --grades {grade} --subject-refs {refs_str} --count 4")
        print(f"```")
        print()

def generate_topic_specific_commands():
    """Generate commands for specific non-diagram topics"""
    print("## 📊 Topic-Specific No-Diagram Commands")
    print()

    topics = {
        "Number Operations": ["C1.1", "C1.2", "C1.4", "C1.6", "C1.8"],
        "Algebra": ["C2.1", "C2.2", "C2.5", "C2.6", "C2.9"],
        "Percentage & Ratio": ["C1.7", "C3.1", "C3.2", "C3.3"],
        "Text-Based Geometry": ["C5.2", "C5.4"],
        "Statistics": ["C7.1", "C7.2", "C7.5", "C8.3"],
        "Probability": ["C9.1", "C9.2", "C9.3"],
        "Extended Number": ["E1.1", "E1.3", "E1.5", "E2.1"],
        "Extended Algebra": ["E2.5", "E2.7", "E2.9", "E3.1"],
        "Extended Statistics": ["E7.1", "E7.3", "E8.1", "E9.1"]
    }

    for topic_name, refs in topics.items():
        refs_str = " ".join(refs)
        grades = "5 6 7 8" if topic_name.startswith("Extended") else "3 4 5 6"

        print(f"### **{topic_name}**")
        print(f"```bash")
        print(f"python main.py generate --grades {grades} --subject-refs {refs_str} --count 3")
        print(f"```")
        print()

def generate_random_sampling_commands():
    """Generate commands that randomly sample from all no-diagram topics"""
    print("## 🎲 Random No-Diagram Sampling Commands")
    print()

    print("### **Random Foundation (Grades 1-6)**")
    foundation_refs = [ref for ref in NO_DIAGRAM_REFS if not ref.startswith("E")]
    sample_foundation = random.sample(foundation_refs, min(10, len(foundation_refs)))
    print(f"```bash")
    print(f"python main.py generate --grades 1 2 3 4 5 6 --subject-refs {' '.join(sample_foundation)} --count 2")
    print(f"```")
    print()

    print("### **Random Higher (Grades 6-9)**")
    all_refs_sample = random.sample(NO_DIAGRAM_REFS, min(12, len(NO_DIAGRAM_REFS)))
    print(f"```bash")
    print(f"python main.py generate --grades 6 7 8 9 --subject-refs {' '.join(all_refs_sample)} --count 2")
    print(f"```")
    print()

    print("### **Random Mixed (All Topics)**")
    mixed_sample = random.sample(NO_DIAGRAM_REFS, min(8, len(NO_DIAGRAM_REFS)))
    print(f"```bash")
    print(f"python main.py generate --grades 4 5 6 7 8 --subject-refs {' '.join(mixed_sample)} --count 3")
    print(f"```")
    print()

def generate_geometry_text_commands():
    """Generate commands for geometry topics that work well with text descriptions"""
    print("## 📐 Text-Based Geometry Commands")
    print("*(Geometry questions using given dimensions, no visual diagrams)*")
    print()

    print("### **Area & Perimeter Calculations**")
    print("```bash")
    print("python main.py generate --grades 4 5 6 --subject-refs C5.2 --count 4")
    print("```")
    print()

    print("### **Volume & Surface Area**")
    print("```bash")
    print("python main.py generate --grades 6 7 8 --subject-refs C5.4 E5.4 --count 3")
    print("```")
    print()

    print("### **Combined Mensuration**")
    print("```bash")
    print("python main.py generate --grades 5 6 7 8 --subject-refs C5.2 C5.4 E5.2 E5.4 --count 3")
    print("```")
    print()

def generate_batch_commands():
    """Generate large batch commands for quick database population"""
    print("## ⚡ Batch No-Diagram Population Commands")
    print()

    print("### **Quick 50 Questions (5 Tabs)**")
    print()

    for i in range(1, 6):
        # Different random samples for each tab
        sample_refs = random.sample(NO_DIAGRAM_REFS, min(8, len(NO_DIAGRAM_REFS)))
        grades = ["1 2 3", "3 4 5", "5 6 7", "6 7 8", "7 8 9"][i-1]

        print(f"**Tab {i}:**")
        print(f"```bash")
        print(f"python main.py generate --grades {grades} --subject-refs {' '.join(sample_refs)} --count 10")
        print(f"```")
        print()

def main():
    """Generate all no-diagram command variations"""
    print("# No-Diagram Question Generation Commands")
    print("*Cambridge IGCSE Mathematics questions that don't require visual diagrams*")
    print()
    print("These commands focus on topics that can be expressed purely through text:")
    print("- Number operations and calculations")
    print("- Algebraic expressions and equations")
    print("- Area/volume calculations with given dimensions")
    print("- Statistics and probability with data tables")
    print("- Coordinate geometry with given coordinates")
    print()

    generate_basic_commands()
    generate_topic_specific_commands()
    generate_random_sampling_commands()
    generate_geometry_text_commands()
    generate_batch_commands()

    print("## 📋 Notes")
    print("- All these topics work well with `raw_text_content` for frontend parsing")
    print("- Questions include dimensions/data in text descriptions")
    print("- No visual diagram generation required")
    print("- Perfect for automated generation and testing")
    print("- Frontend can render these as pure text or simple formatted text")

if __name__ == "__main__":
    main()
