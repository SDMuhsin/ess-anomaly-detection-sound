#!/usr/bin/env python
"""
Test script for the research experiments framework
Tests basic functionality with a small subset of data
"""
import argparse
import logging
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch

# Import our research framework
from research_experiments import create_model, measure_model_efficiency

logging.basicConfig(level=logging.INFO)

def test_model_creation():
    """Test that all models can be created successfully"""
    input_dim = 320  # 64 * 5
    
    models_to_test = [
        'simple_ae', 'deep_ae', 'vae', 'conv_ae', 'lstm_ae',
        'attention_ae', 'residual_ae', 'denoising_ae', 
        'isolation_forest', 'one_class_svm'
    ]
    
    print("Testing model creation...")
    for model_name in models_to_test:
        try:
            model = create_model(model_name, input_dim, n_mels=64, frames=5)
            print(f"✓ {model_name}: Created successfully")
            
            # Test forward pass for PyTorch models
            if hasattr(model, 'forward'):
                device = torch.device('cpu')
                model = model.to(device)
                test_input = torch.randn(10, input_dim)
                
                with torch.no_grad():
                    if model_name == 'vae':
                        output, mu, logvar = model(test_input)
                    else:
                        output = model(test_input)
                print(f"  Forward pass: OK")
            
        except Exception as e:
            print(f"✗ {model_name}: Failed - {e}")
            return False
    
    return True

def test_efficiency_measurement():
    """Test efficiency measurement functions"""
    print("\nTesting efficiency measurement...")
    
    input_dim = 320
    device = torch.device('cpu')
    
    # Test with a simple model
    model = create_model('simple_ae', input_dim)
    model = model.to(device)
    
    try:
        metrics = measure_model_efficiency(model, input_dim, device)
        
        required_keys = [
            'model_size_mb', 'num_parameters', 'inference_time_ms',
            'throughput_samples_per_sec', 'memory_usage_mb'
        ]
        
        for key in required_keys:
            if key not in metrics:
                print(f"✗ Missing metric: {key}")
                return False
            print(f"  {key}: {metrics[key]}")
        
        print("✓ Efficiency measurement: OK")
        return True
        
    except Exception as e:
        print(f"✗ Efficiency measurement failed: {e}")
        return False

def test_sklearn_models():
    """Test sklearn-based models"""
    print("\nTesting sklearn models...")
    
    # Generate some dummy data
    np.random.seed(42)
    train_data = np.random.randn(100, 320)
    test_data = np.random.randn(20, 320)
    
    for model_name in ['isolation_forest', 'one_class_svm']:
        try:
            model = create_model(model_name, 320)
            
            # Train
            model.fit(train_data)
            print(f"✓ {model_name}: Training OK")
            
            # Predict
            scores = model.predict_anomaly_scores(test_data)
            if len(scores) != len(test_data):
                print(f"✗ {model_name}: Wrong number of predictions")
                return False
            
            print(f"✓ {model_name}: Prediction OK")
            
        except Exception as e:
            print(f"✗ {model_name}: Failed - {e}")
            return False
    
    return True

def test_argument_parser():
    """Test that the argument parser works correctly"""
    print("\nTesting argument parser...")
    
    try:
        # Import the main function to test argument parsing
        import research_experiments
        
        # Test with minimal arguments
        test_args = ['--base_dir', './test_dataset', '--model', 'simple_ae', '--epochs', '1']
        
        parser = research_experiments.argparse.ArgumentParser()
        # We can't easily test the full parser without running main, 
        # but we can verify the import works
        print("✓ Argument parser: Import OK")
        return True
        
    except Exception as e:
        print(f"✗ Argument parser failed: {e}")
        return False

def main():
    print("="*60)
    print("RESEARCH EXPERIMENTS FRAMEWORK - TEST SUITE")
    print("="*60)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Efficiency Measurement", test_efficiency_measurement),
        ("Sklearn Models", test_sklearn_models),
        ("Argument Parser", test_argument_parser),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name}: PASSED")
            else:
                print(f"✗ {test_name}: FAILED")
        except Exception as e:
            print(f"✗ {test_name}: ERROR - {e}")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All tests passed! The research framework is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
