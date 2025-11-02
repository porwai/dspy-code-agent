#!/usr/bin/env python3
"""Simple test script to verify MLflow trace tag setting works."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from mlflow.tracking import MlflowClient
import time

# Enable DSPy autolog
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("test_tags")

try:
    if hasattr(mlflow, "dspy") and hasattr(mlflow.dspy, "autolog"):
        mlflow.dspy.autolog()
        print("✓ MLflow DSPy autolog enabled")
except Exception as e:
    print(f"Warning: Could not enable autolog: {e}")

# Test 1: Try update_current_trace within a traced function
print("\n=== Test 1: update_current_trace within trace ===")
try:
    @mlflow.trace
    def test_function_1():
        mlflow.update_current_trace(tags={"test_tag_1": "value_1", "test_instance_id": "test-123"})
        return "success"
    
    result = test_function_1()
    print(f"✓ Function executed: {result}")
    
    # Try to get trace ID
    trace_id = None
    try:
        if hasattr(mlflow, "get_last_active_trace_id"):
            trace_id = mlflow.get_last_active_trace_id()
            print(f"✓ Got trace ID: {trace_id}")
    except Exception as e:
        print(f"✗ Could not get trace ID: {e}")
    
    # Wait a bit for trace to be committed
    time.sleep(1)
    
    # Verify tags were set
    if trace_id:
        client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
        trace = client.get_trace(trace_id=trace_id)
        tags = getattr(trace.info, "tags", {}) or {}
        print(f"✓ Trace tags: {tags}")
        if "test_tag_1" in tags:
            print(f"✓ Tag 'test_tag_1' found: {tags['test_tag_1']}")
        else:
            print(f"✗ Tag 'test_tag_1' NOT found in tags: {list(tags.keys())}")
        if "test_instance_id" in tags:
            print(f"✓ Tag 'test_instance_id' found: {tags['test_instance_id']}")
        else:
            print(f"✗ Tag 'test_instance_id' NOT found")
except Exception as e:
    print(f"✗ Test 1 failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Try set_trace_tag after trace completes
print("\n=== Test 2: set_trace_tag after trace ===")
try:
    @mlflow.trace
    def test_function_2():
        return "success"
    
    result = test_function_2()
    print(f"✓ Function executed: {result}")
    
    # Get trace ID
    trace_id = None
    try:
        if hasattr(mlflow, "get_last_active_trace_id"):
            trace_id = mlflow.get_last_active_trace_id()
            print(f"✓ Got trace ID: {trace_id}")
    except Exception as e:
        print(f"✗ Could not get trace ID: {e}")
    
    # Wait a bit
    time.sleep(1)
    
    # Try to set tags via client
    if trace_id:
        client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
        try:
            client.set_trace_tag(trace_id, "test_tag_2", "value_2")
            client.set_trace_tag(trace_id, "instance_id", "test-456")
            print(f"✓ Set tags via client.set_trace_tag()")
        except Exception as e:
            print(f"✗ Failed to set tags: {e}")
        
        # Verify
        trace = client.get_trace(trace_id=trace_id)
        tags = getattr(trace.info, "tags", {}) or {}
        print(f"✓ Trace tags: {tags}")
        if "test_tag_2" in tags:
            print(f"✓ Tag 'test_tag_2' found: {tags['test_tag_2']}")
        else:
            print(f"✗ Tag 'test_tag_2' NOT found in tags: {list(tags.keys())}")
        if "instance_id" in tags:
            print(f"✓ Tag 'instance_id' found: {tags['instance_id']}")
        else:
            print(f"✗ Tag 'instance_id' NOT found")
except Exception as e:
    print(f"✗ Test 2 failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Summary ===")
print("Check MLflow UI at http://127.0.0.1:5000 to verify tags are visible")

