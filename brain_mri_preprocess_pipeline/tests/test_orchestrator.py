#!/usr/bin/env python3
"""
Unit tests for the pipeline orchestrator.
"""

import unittest
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from brain_mri_preprocess_pipeline.utils.pipeline_orchestrator import PipelineOrchestrator, PipelineStep
from brain_mri_preprocess_pipeline.utils.models import ProcessingStats
from brain_mri_preprocess_pipeline.utils.logging_utils import LogManager
from brain_mri_preprocess_pipeline.utils.config import PipelineConfig

class MockProcessor:
    """Mock processor for testing"""
    
    def __init__(self, name: str, success_rate: float = 1.0):
        """Initialize the mock processor"""
        self.name = name
        self.success_rate = success_rate
        self.logger = MagicMock()
    
    def process_directory(self, parallel: bool = False, max_workers: int = 4) -> ProcessingStats:
        """Mock processing method"""
        # Create mock statistics
        stats = ProcessingStats()
        stats.total_cases = 10
        stats.successful_cases = int(10 * self.success_rate)
        stats.failed_cases = stats.total_cases - stats.successful_cases
        stats.processing_time = 1.0
        
        return stats

class TestPipelineOrchestrator(unittest.TestCase):
    """Tests for the PipelineOrchestrator class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "input"
        self.output_dir = Path(self.temp_dir) / "output"
        self.log_dir = Path(self.temp_dir) / "logs"
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        self.log_dir.mkdir()
        
        # Create configuration
        self.config = PipelineConfig(
            input_directory=self.input_dir,
            output_directory=self.output_dir,
            log_directory=self.log_dir,
            parallel_workers=2
        )
        
        # Create log manager
        self.log_manager = MagicMock()
        
        # Create orchestrator
        self.orchestrator = PipelineOrchestrator(self.config, self.log_manager)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test initialization"""
        self.assertEqual(self.orchestrator.config, self.config)
        self.assertEqual(self.orchestrator.log_manager, self.log_manager)
        self.assertEqual(len(self.orchestrator.steps), 0)
    
    def test_add_step(self):
        """Test adding a step to the pipeline"""
        # Create mock processor
        processor = MockProcessor("test_processor")
        
        # Create step
        step = PipelineStep(
            name="test_step",
            description="Test step",
            processor=processor,
            method_name="process_directory"
        )
        
        # Add step to orchestrator
        self.orchestrator.add_step(step)
        
        # Verify results
        self.assertEqual(len(self.orchestrator.steps), 1)
        self.assertEqual(self.orchestrator.steps[0], step)
    
    def test_execute_pipeline_empty(self):
        """Test executing an empty pipeline"""
        # Execute pipeline
        success = self.orchestrator.execute_pipeline()
        
        # Verify results
        self.assertTrue(success)
    
    def test_execute_pipeline_success(self):
        """Test executing a pipeline with all successful steps"""
        # Create mock processors
        processors = [
            MockProcessor("processor1", 1.0),
            MockProcessor("processor2", 1.0),
            MockProcessor("processor3", 1.0)
        ]
        
        # Add steps to orchestrator
        for i, processor in enumerate(processors):
            step = PipelineStep(
                name=f"step{i+1}",
                description=f"Step {i+1}",
                processor=processor,
                method_name="process_directory"
            )
            self.orchestrator.add_step(step)
        
        # Execute pipeline
        success = self.orchestrator.execute_pipeline()
        
        # Verify results
        self.assertTrue(success)
    
    def test_execute_pipeline_with_failures(self):
        """Test executing a pipeline with some failing steps"""
        # Create mock processors with varying success rates
        processors = [
            MockProcessor("processor1", 1.0),
            MockProcessor("processor2", 0.5),  # 50% success rate
            MockProcessor("processor3", 1.0)
        ]
        
        # Add steps to orchestrator
        for i, processor in enumerate(processors):
            step = PipelineStep(
                name=f"step{i+1}",
                description=f"Step {i+1}",
                processor=processor,
                method_name="process_directory"
            )
            self.orchestrator.add_step(step)
        
        # Execute pipeline
        success = self.orchestrator.execute_pipeline()
        
        # Verify results
        # Pipeline should still complete even with partial failures in steps
        self.assertTrue(success)
    
    def test_execute_pipeline_parallel(self):
        """Test executing a pipeline in parallel"""
        # Create mock processors
        processors = [
            MockProcessor("processor1", 1.0),
            MockProcessor("processor2", 1.0),
            MockProcessor("processor3", 1.0)
        ]
        
        # Add steps to orchestrator
        for i, processor in enumerate(processors):
            step = PipelineStep(
                name=f"step{i+1}",
                description=f"Step {i+1}",
                processor=processor,
                method_name="process_directory"
            )
            self.orchestrator.add_step(step)
        
        # Execute pipeline in parallel
        success = self.orchestrator.execute_pipeline(parallel=True)
        
        # Verify results
        self.assertTrue(success)
    
    def test_get_step_by_name(self):
        """Test getting a step by name"""
        # Create mock processor
        processor = MockProcessor("test_processor")
        
        # Create step
        step = PipelineStep(
            name="test_step",
            description="Test step",
            processor=processor,
            method_name="process_directory"
        )
        
        # Add step to orchestrator
        self.orchestrator.add_step(step)
        
        # Get step by name
        retrieved_step = self.orchestrator.get_step_by_name("test_step")
        
        # Verify results
        self.assertEqual(retrieved_step, step)
    
    def test_get_step_by_name_not_found(self):
        """Test getting a nonexistent step by name"""
        # Get step by name
        retrieved_step = self.orchestrator.get_step_by_name("nonexistent_step")
        
        # Verify results
        self.assertIsNone(retrieved_step)

if __name__ == "__main__":
    unittest.main()