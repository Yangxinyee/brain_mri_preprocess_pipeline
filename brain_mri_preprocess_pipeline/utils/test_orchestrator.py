#!/usr/bin/env python3
"""
Test script for the pipeline orchestrator.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path to allow importing from sibling modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.pipeline_orchestrator import PipelineOrchestrator, PipelineStep
from utils.logging_utils import LogManager
from utils.config import PipelineConfig
from utils.models import ProcessingStats

class MockProcessor:
    """Mock processor for testing"""
    
    def __init__(self, name: str, success_rate: float = 1.0):
        """Initialize the mock processor"""
        self.name = name
        self.success_rate = success_rate
        self.logger = logging.getLogger(name)
    
    def process_directory(self, parallel: bool = False, max_workers: int = 4) -> ProcessingStats:
        """Mock processing method"""
        self.logger.info(f"Processing with {self.name}")
        
        # Create mock statistics
        stats = ProcessingStats()
        stats.total_cases = 10
        stats.successful_cases = int(10 * self.success_rate)
        stats.failed_cases = stats.total_cases - stats.successful_cases
        stats.processing_time = 1.0
        
        return stats

def main():
    """Main function"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create log directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create configuration
    config = PipelineConfig(
        input_directory=Path("test_input"),
        output_directory=Path("test_output"),
        log_directory=log_dir,
        parallel_workers=2
    )
    
    # Create log manager
    log_manager = LogManager(config.log_directory)
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(config, log_manager)
    
    # Create mock processors
    processors = [
        MockProcessor("decompression", 1.0),
        MockProcessor("conversion", 0.9),
        MockProcessor("registration", 0.8),
        MockProcessor("skull_stripping", 0.7),
        MockProcessor("encryption", 1.0),
        MockProcessor("organization", 0.9)
    ]
    
    # Add steps to orchestrator
    for processor in processors:
        orchestrator.add_step(PipelineStep(
            name=processor.name,
            description=f"Mock {processor.name} step",
            processor=processor,
            method_name="process_directory"
        ))
    
    # Execute pipeline
    success = orchestrator.execute_pipeline(parallel=True)
    
    print(f"Pipeline execution {'succeeded' if success else 'failed'}")

if __name__ == "__main__":
    main()