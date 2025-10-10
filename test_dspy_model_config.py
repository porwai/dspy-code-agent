#!/usr/bin/env python3
"""Test script to verify DSPy agent works with different model providers."""

import os
from pathlib import Path
from minisweagent.agents.dspy import DSPyAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model

def test_dspy_with_different_models():
    """Test DSPy agent with different model configurations."""
    
    # Test 1: OpenAI model
    print("Testing with OpenAI model...")
    try:
        model = get_model("gpt-4o-mini")
        env = LocalEnvironment()
        agent = DSPyAgent(model, env)
        print("✓ OpenAI model configuration successful")
    except Exception as e:
        print(f"✗ OpenAI model failed: {e}")
    
    # Test 2: Anthropic model (if API key available)
    print("\nTesting with Anthropic model...")
    try:
        if os.getenv('ANTHROPIC_API_KEY'):
            model = get_model("claude-3-5-sonnet-20241022")
            env = LocalEnvironment()
            agent = DSPyAgent(model, env)
            print("✓ Anthropic model configuration successful")
        else:
            print("⚠ Anthropic API key not found, skipping test")
    except Exception as e:
        print(f"✗ Anthropic model failed: {e}")
    
    # Test 3: OpenRouter model (if API key available)
    print("\nTesting with OpenRouter model...")
    try:
        if os.getenv('OPENROUTER_API_KEY'):
            model = get_model("anthropic/claude-3-5-sonnet", model_class="openrouter")
            env = LocalEnvironment()
            agent = DSPyAgent(model, env)
            print("✓ OpenRouter model configuration successful")
        else:
            print("⚠ OpenRouter API key not found, skipping test")
    except Exception as e:
        print(f"✗ OpenRouter model failed: {e}")
    
    # Test 4: LiteLLM model
    print("\nTesting with LiteLLM model...")
    try:
        model = get_model("gpt-4o-mini", model_class="litellm")
        env = LocalEnvironment()
        agent = DSPyAgent(model, env)
        print("✓ LiteLLM model configuration successful")
    except Exception as e:
        print(f"✗ LiteLLM model failed: {e}")

if __name__ == "__main__":
    test_dspy_with_different_models()
