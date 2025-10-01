#!/usr/bin/env python3

"""Test script to verify DSPy agent integration."""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from minisweagent.agents.dspy import DSPyAgent, DSPyAgentConfig
        print("✓ DSPy agent imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    try:
        from minisweagent.environments.local import LocalEnvironment
        print("✓ Environment import successful")
    except ImportError as e:
        print(f"✗ Environment import error: {e}")
        return False
    
    try:
        from minisweagent.models.litellm_model import LitellmModel
        print("✓ Model import successful")
    except ImportError as e:
        print(f"✗ Model import error: {e}")
        return False
    
    return True

def test_agent_creation():
    """Test that the agent can be created."""
    print("\nTesting agent creation...")
    
    try:
        from minisweagent.agents.dspy import DSPyAgent, DSPyAgentConfig
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.litellm_model import LitellmModel
        
        # Create components
        model = LitellmModel(model_name="gpt-4o")
        env = LocalEnvironment()
        
        # Create agent
        agent = DSPyAgent(model=model, env=env)
        print("✓ DSPy agent created successfully")
        
        # Test configuration
        assert hasattr(agent, 'config')
        assert hasattr(agent, 'model')
        assert hasattr(agent, 'env')
        assert hasattr(agent, 'messages')
        print("✓ Agent has required attributes")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent creation error: {e}")
        return False

def test_framework_compatibility():
    """Test that the agent follows framework conventions."""
    print("\nTesting framework compatibility...")
    
    try:
        from minisweagent.agents.dspy import DSPyAgent, DSPyAgentConfig
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.litellm_model import LitellmModel
        
        model = LitellmModel(model_name="gpt-4o")
        env = LocalEnvironment()
        agent = DSPyAgent(model=model, env=env)
        
        # Test required methods exist
        assert hasattr(agent, 'run')
        assert hasattr(agent, 'step')
        assert hasattr(agent, 'add_message')
        print("✓ Agent has required methods")
        
        # Test method signatures
        import inspect
        
        run_sig = inspect.signature(agent.run)
        assert 'task' in run_sig.parameters
        print("✓ run() method has correct signature")
        
        step_sig = inspect.signature(agent.step)
        print("✓ step() method has correct signature")
        
        add_msg_sig = inspect.signature(agent.add_message)
        assert 'role' in add_msg_sig.parameters
        assert 'content' in add_msg_sig.parameters
        print("✓ add_message() method has correct signature")
        
        return True
        
    except Exception as e:
        print(f"✗ Framework compatibility error: {e}")
        return False

def main():
    """Run all tests."""
    print("DSPy Agent Integration Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_agent_creation,
        test_framework_compatibility,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! DSPy agent integration is working.")
    else:
        print("❌ Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
