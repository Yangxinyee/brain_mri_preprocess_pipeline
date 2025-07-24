#!/usr/bin/env python3
"""
Test runner for the brain MRI preprocessing pipeline.
Runs all unit tests and reports results.
"""

import unittest
import sys
import argparse
from pathlib import Path

# Add parent directory to path to allow importing from parent modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import test modules
from brain_mri_preprocess_pipeline.tests.test_models import TestProcessingStats, TestModalityStatus, TestCaseInfo, TestPipelineError, TestErrorHandler, TestChannelMapping
from brain_mri_preprocess_pipeline.tests.test_decompressor import TestDicomDecompressor
from brain_mri_preprocess_pipeline.tests.test_converter import TestDicomToNiftiConverter
from brain_mri_preprocess_pipeline.tests.test_registration import TestImageRegistrationEngine
from brain_mri_preprocess_pipeline.tests.test_skull_stripper import TestSkullStrippingProcessor
from brain_mri_preprocess_pipeline.tests.test_orchestrator import TestPipelineOrchestrator
from brain_mri_preprocess_pipeline.tests.test_integration import TestIntegrationPipeline, TestIntegrationValidation

def run_tests(test_type="all"):
    """
    Run tests based on the specified type
    
    Args:
        test_type: Type of tests to run ("unit", "integration", or "all")
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Create test suite
    test_suite = unittest.TestSuite()
    
    if test_type in ["unit", "all"]:
        print("Running unit tests...")
        
        # Add test cases for models
        test_suite.addTest(unittest.makeSuite(TestProcessingStats))
        test_suite.addTest(unittest.makeSuite(TestModalityStatus))
        test_suite.addTest(unittest.makeSuite(TestCaseInfo))
        test_suite.addTest(unittest.makeSuite(TestPipelineError))
        test_suite.addTest(unittest.makeSuite(TestErrorHandler))
        test_suite.addTest(unittest.makeSuite(TestChannelMapping))
        
        # Add test cases for modules
        test_suite.addTest(unittest.makeSuite(TestDicomDecompressor))
        test_suite.addTest(unittest.makeSuite(TestDicomToNiftiConverter))
        test_suite.addTest(unittest.makeSuite(TestImageRegistrationEngine))
        test_suite.addTest(unittest.makeSuite(TestSkullStrippingProcessor))
        test_suite.addTest(unittest.makeSuite(TestPipelineOrchestrator))
    
    if test_type in ["integration", "all"]:
        print("Running integration tests...")
        
        # Add integration test cases
        test_suite.addTest(unittest.makeSuite(TestIntegrationPipeline))
        test_suite.addTest(unittest.makeSuite(TestIntegrationValidation))
    
    # Run tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run tests for the brain MRI preprocessing pipeline")
    parser.add_argument("--type", choices=["unit", "integration", "all"], default="all",
                        help="Type of tests to run (default: all)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run performance benchmarking tests")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.benchmark:
        # Run only performance benchmarking tests
        test_suite = unittest.TestSuite()
        test_suite.addTest(unittest.makeSuite(TestIntegrationPipeline))
        
        # Filter for benchmark tests
        for test in test_suite:
            for test_case in test:
                if not test_case._testMethodName.startswith('test_performance'):
                    setattr(test_case, test_case._testMethodName, lambda: None)
        
        # Run tests
        test_runner = unittest.TextTestRunner(verbosity=2)
        result = test_runner.run(test_suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    else:
        # Run regular tests
        sys.exit(run_tests(args.type))