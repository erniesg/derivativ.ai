# Installation Instructions

## Quick Setup

1. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

2. **Set up API keys (optional for testing):**
   ```bash
   python tools/setup_api_keys.py
   ```

3. **Run examples:**
   ```bash
   python examples/mock_workflow.py
   python examples/live_apis.py
   ```

4. **Run tests:**
   ```bash
   pytest tests/ -v
   ```

## What the `pip install -e .` does

- Installs the `derivativ` package in "editable" mode
- Allows importing from `src/` using normal Python imports
- No more `sys.path` manipulation needed
- All examples and tests work with clean imports

## Package Structure

```
derivativ.ai/
├── src/                    # Main package code
├── tests/                  # Test suite
├── examples/               # Demo scripts
├── tools/                  # Setup utilities
└── setup.py               # Package configuration
```
