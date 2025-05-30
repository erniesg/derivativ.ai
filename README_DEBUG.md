# Debug Configuration

The IGCSE Question Generation System now includes configurable debug output to help troubleshoot issues with LLM responses and JSON parsing.

## Enabling Debug Mode

### Method 1: Environment Variable
Set the `DEBUG` environment variable:

```bash
# Enable debug mode
export DEBUG=true
python main.py generate --grades 5 --count 1

# Or inline
DEBUG=true python main.py generate --grades 5 --count 1
```

### Method 2: CLI Flag
Use the `--debug` flag with any command:

```bash
# Enable debug for generation
python main.py --debug generate --grades 5 --count 1

# Enable debug for listing
python main.py --debug list --grade 5

# Enable debug for stats
python main.py --debug stats
```

### Method 3: .env File
Add to your `.env` file:

```bash
DEBUG=true
```

## Debug Output

When debug mode is enabled, you'll see detailed information about:

### 1. LLM Calls
```
[DEBUG] =================== LLM CALL START ===================
[DEBUG] Model: Qwen/Qwen3-235B-A22B
[DEBUG] Temperature: 0.7
[DEBUG] Max tokens: 4000
[DEBUG] Prompt length: 3247 characters
[DEBUG] Prompt start: You are an expert Cambridge IGCSE Mathematics...
[DEBUG] Prompt end: ...NO CODE BLOCKS, NO THINKING TAGS, NO EXTRA TEXT.**
[DEBUG] Calling model with 2 messages
```

### 2. LLM Responses
```
[DEBUG] =================== LLM RESPONSE ===================
[DEBUG] Response type: <class 'str'>
[DEBUG] Response length: 1234 characters
[DEBUG] Full response:
{
  "question_id_local": "Gen_Q1234",
  "question_id_global": "gen_abc123_q567",
  ...
}
[DEBUG] =================== END RESPONSE ===================
```

### 3. JSON Parsing
```
[DEBUG] =================== JSON PARSING START ===================
[DEBUG] Original response length: 1234 characters
[DEBUG] After stripping thinking tokens: 1234 characters
[DEBUG] Attempting to extract JSON from code blocks...
[DEBUG] No JSON found in code blocks, trying raw JSON extraction...
[DEBUG] ✅ Successfully extracted raw JSON!
[DEBUG] Extracted JSON keys: ['question_id_local', 'question_id_global', ...]
[DEBUG] =================== JSON PARSING END ===================
```

### 4. Field Validation
```
[DEBUG] =================== FIELD VALIDATION START ===================
[DEBUG] Parsed question_data keys: ['question_id_local', 'question_id_global', ...]
[DEBUG] ✅ All top-level fields present
[DEBUG] Taxonomy keys: ['topic_path', 'subject_content_references', ...]
[DEBUG] Solution keys: ['final_answers_summary', 'mark_allocation_criteria', ...]
[DEBUG] Solver keys: ['steps']
[DEBUG] Question text length: 87
[DEBUG] Marks: 1
[DEBUG] Number of answers: 1
[DEBUG] Number of criteria: 1
[DEBUG] Number of steps: 1
[DEBUG] =================== FIELD VALIDATION END ===================
```

## Accepted Values

The `DEBUG` environment variable accepts these values (case-insensitive):

**Enable Debug:** `true`, `1`, `yes`, `on`
**Disable Debug:** `false`, `0`, `no`, `off`, or any other value

## Troubleshooting with Debug Mode

### Common Issues Revealed by Debug Output:

1. **Missing JSON Fields**: Look for "❌ Missing required fields" messages
2. **Thinking Tokens**: Check if `<think>...</think>` blocks are being stripped
3. **Code Block Parsing**: See if JSON is wrapped in ```json blocks
4. **LLM Response Format**: Examine the full response to understand output structure
5. **Field Validation**: Check which nested fields are missing or malformed

### Example Debug Session:

```bash
# Test with Hugging Face model and debug enabled
DEBUG=true python main.py generate --grades 5 --model "Qwen/Qwen3-235B-A22B" --count 1

# This will show you exactly:
# - What prompt is sent to the model
# - What response comes back
# - How JSON parsing works (or fails)
# - Which fields are present/missing
# - Any validation errors
```

## Performance Note

Debug mode generates significant output and may impact performance. **Only enable it when troubleshooting issues.**

For normal usage, keep debug mode disabled:

```bash
# Normal usage (no debug output)
python main.py generate --grades 5 --count 1
```
