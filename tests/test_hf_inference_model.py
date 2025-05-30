#!/usr/bin/env python3
"""
Test script for Hugging Face InferenceClientModel integration with Gemma (serverless), DeepSeek, and Qwen3 (auto provider).
"""

import os
from dotenv import load_dotenv
from smolagents import InferenceClientModel

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

MODELS = [
    ("deepseek-ai/DeepSeek-R1-0528", "DeepSeek-R1-0528", "auto", None),
    ("Qwen/Qwen3-235B-A22B", "Qwen3-235B-A22B", "auto", None)
]

PROMPT = """Generate a simple math question in JSON format:\n{\n  \"question\": \"What is 11 + 6?\",\n  \"answer\": 17,\n  \"difficulty\": \"easy\"\n}"""

messages = [
    {"role": "user", "content": PROMPT}
]

def test_model(model_id, model_name, provider, bill_to):
    print(f"\n=== Testing {model_name} ({model_id}) with provider '{provider}'" + (f" (billed to {bill_to})" if bill_to else "") + " ===")
    if not HF_TOKEN:
        print("❌ HF_TOKEN not found in environment")
        return
    try:
        model_kwargs = {
            "model_id": model_id,
            "provider": provider,
            "token": HF_TOKEN,
            "max_tokens": 1000
        }

        if bill_to:
            model_kwargs["bill_to"] = bill_to

        model = InferenceClientModel(**model_kwargs)
        response = model(messages)
        print(f"Raw response type: {type(response)}")
        print(f"Raw response:\n{response}\n")
        content = getattr(response, "content", response)
        print(f"Extracted content:\n{content}\n")
    except Exception as e:
        print(f"❌ Error with {model_name}: {e}")

if __name__ == "__main__":
    for model_id, model_name, provider, bill_to in MODELS:
        test_model(model_id, model_name, provider, bill_to)
