# Pandoc Setup Guide for Document Generation

This guide explains how to install and configure pandoc for the document generation system.

## 🎯 What is Pandoc?

Pandoc is a universal document converter that can transform documents between numerous markup and word processing formats. Our system uses it to convert Markdown documents to:

- **PDF** (via LaTeX)
- **DOCX** (Microsoft Word)
- **PPTX** (PowerPoint presentations)
- **LaTeX** (academic formatting)
- **HTML** (web pages)

## 📦 Installation

### macOS (recommended)

```bash
# Install pandoc
brew install pandoc

# Install LaTeX for PDF generation (large download ~4GB)
brew install --cask mactex

# Verify installation
pandoc --version
pdflatex --version
```

### Ubuntu/Debian

```bash
# Install pandoc
sudo apt-get update
sudo apt-get install pandoc

# Install LaTeX for PDF generation
sudo apt-get install texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended

# Verify installation
pandoc --version
pdflatex --version
```

### Windows

```bash
# Using Chocolatey
choco install pandoc
choco install miktex

# Using Scoop
scoop install pandoc
scoop install latex
```

### Alternative: Pandoc Binary Only

If you don't need PDF generation, you can install just pandoc:

```bash
# This will work for HTML, DOCX, PPTX, LaTeX output
# PDF generation will fail without LaTeX
brew install pandoc  # macOS
sudo apt install pandoc  # Ubuntu
```

## 🔧 Configuration

### Environment Variables

Add to your `.env` file:

```bash
# Optional: Custom pandoc path
PANDOC_PATH=/usr/local/bin/pandoc

# Optional: Custom LaTeX path  
PDFLATEX_PATH=/usr/local/bin/pdflatex

# Optional: Enable/disable specific formats
ENABLE_PDF_GENERATION=true
ENABLE_DOCX_GENERATION=true
ENABLE_SLIDES_GENERATION=true
```

### Template Customization

You can customize document templates by placing them in `templates/pandoc/`:

```
templates/pandoc/
├── reference.docx          # Word document styling
├── reference.pptx          # PowerPoint template
├── latex-template.tex      # LaTeX template
└── css/
    └── styles.css          # HTML styling
```

## 🧪 Testing Your Installation

Run our demo script to verify everything works:

```bash
# Test all formats
python examples/demo_pandoc_conversion.py

# Check what's available
pandoc --list-output-formats
```

## ⚡ Quick Test

Create a simple test:

```bash
# Create test markdown
echo "# Test Document\n\nThis is a **test**." > test.md

# Convert to different formats
pandoc test.md -o test.html
pandoc test.md -o test.docx
pandoc test.md -o test.pdf    # Requires LaTeX
pandoc test.md -o test.pptx -t pptx

# Check results
ls -la test.*
```

## 🐛 Troubleshooting

### Common Issues

#### 1. "pandoc: command not found"
```bash
# Check if pandoc is in PATH
which pandoc
echo $PATH

# If not found, add to PATH or reinstall
```

#### 2. "pandoc: pdflatex not found"
```bash
# Install LaTeX distribution
brew install --cask mactex     # macOS
sudo apt install texlive-full  # Ubuntu (large download)

# Check LaTeX is working
pdflatex --version
```

#### 3. "pandoc: Could not find reference.docx"
```bash
# This warning is safe to ignore, or create a custom reference doc:
pandoc --print-default-data-file reference.docx > reference.docx
```

#### 4. Memory issues with large documents
```bash
# For very large documents, increase memory:
pandoc +RTS -M512m -RTS large-doc.md -o large-doc.pdf
```

### Performance Tips

1. **PDF Generation**: Use `--pdf-engine=xelatex` for Unicode support
2. **Large Documents**: Enable `--toc` for table of contents
3. **Math**: Use `--mathjax` for web math or `--mathml` for Word
4. **Citations**: Add `--citeproc` for bibliography support

## 📊 Format Support Matrix

| Format | Extension | Requirements | Notes |
|--------|-----------|--------------|-------|
| HTML | `.html` | pandoc only | Always works |
| DOCX | `.docx` | pandoc only | Microsoft Word format |
| PDF | `.pdf` | pandoc + LaTeX | Best quality, requires large install |
| PPTX | `.pptx` | pandoc only | PowerPoint presentations |
| LaTeX | `.tex` | pandoc only | Academic formatting |
| EPUB | `.epub` | pandoc only | E-book format |

## 🚀 Advanced Usage

### Custom Templates

```bash
# Create custom LaTeX template
pandoc -D latex > custom-template.tex
# Edit custom-template.tex
pandoc document.md --template=custom-template.tex -o document.pdf
```

### Metadata Integration

```yaml
---
title: "My Document"
author: "Derivativ AI"
date: "2025-06-21"
geometry: margin=2cm
fontsize: 12pt
---

# Content starts here
```

### Batch Conversion

```bash
# Convert all markdown files to PDF
for f in *.md; do pandoc "$f" -o "${f%.md}.pdf"; done
```

## 📚 Resources

- [Pandoc User Guide](https://pandoc.org/MANUAL.html)
- [LaTeX Installation Guide](https://www.latex-project.org/get/)
- [Pandoc Templates](https://github.com/jgm/pandoc-templates)
- [Math in Pandoc](https://pandoc.org/MANUAL.html#math)

## ✅ Next Steps

1. Install pandoc and LaTeX following the guide above
2. Run `python examples/demo_pandoc_conversion.py` to test
3. Try generating documents through the API endpoints
4. Customize templates for your organization's branding

The document generation system will automatically detect pandoc and enable the appropriate export formats based on what's installed.