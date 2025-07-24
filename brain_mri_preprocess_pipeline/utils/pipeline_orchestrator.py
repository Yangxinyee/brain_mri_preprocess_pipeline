#!/usr/bin/env python3
"""
Pipeline Orchestrator for the medical image processing pipeline.
Coordinates all processing steps and handles progress tracking and reporting.
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import shutil

from .logging_utils import LogManager
from .config import PipelineConfig
from .models import ProcessingStats, PipelineError, ErrorHandler

@dataclass
class PipelineStep:
    """Class representing a pipeline processing step"""
    name: str
    description: str
    processor: Any
    method_name: str
    enabled: bool = True
    stats: ProcessingStats = field(default_factory=ProcessingStats)
    
    def execute(self, *args, **kwargs) -> ProcessingStats:
        """Execute the step's processing method"""
        if not self.enabled:
            return ProcessingStats()
        
        # Get the processing method
        method = getattr(self.processor, self.method_name)
        
        # Execute the method
        return method(*args, **kwargs)

class PipelineOrchestrator:
    """Coordinates all processing steps in the medical image pipeline"""
    
    def __init__(self, config: PipelineConfig, log_manager: LogManager):
        """
        Initialize the pipeline orchestrator
        
        Args:
            config: Pipeline configuration
            log_manager: Logging manager
        """
        self.config = config
        self.log_manager = log_manager
        self.logger = log_manager.setup_step_logger("orchestrator")
        self.steps: List[PipelineStep] = []
        self.overall_stats = ProcessingStats()
        self.start_time = 0
        self.end_time = 0
        
        self.logger.info("Pipeline Orchestrator initialized")
    
    def add_step(self, step: PipelineStep) -> None:
        """
        Add a processing step to the pipeline
        
        Args:
            step: The processing step to add
        """
        self.steps.append(step)
        self.logger.info(f"Added step: {step.name} - {step.description}")
    
    def execute_pipeline(self, parallel: bool = False) -> bool:
        """
        Execute all pipeline steps
        
        Args:
            parallel: Whether to process cases in parallel within each step
            
        Returns:
            True if all steps completed successfully, False otherwise
        """
        self.start_time = time.time()
        self.logger.info("Starting pipeline execution")
        self.logger.info(f"Parallel processing: {parallel}")
        
        success = True
        total_steps = len(self.steps)
        completed_steps = 0
        
        # Execute each step in sequence
        for i, step in enumerate(self.steps, 1):
            if not step.enabled:
                self.logger.info(f"Skipping disabled step {i}/{total_steps}: {step.name}")
                continue
                
            self.logger.info(f"Executing step {i}/{total_steps}: {step.name} - {step.description}")
            
            try:
                # Execute the step with parallel processing if enabled
                step.stats = step.execute(
                    parallel=(parallel and self.config.parallel_workers > 1),
                    max_workers=self.config.parallel_workers
                )
                
                # Log step completion
                self.logger.info(f"Step {i}/{total_steps} completed: {step.name}")
                self.logger.info(f"  Success rate: {step.stats.successful_cases}/{step.stats.total_cases} cases")
                self.logger.info(f"  Processing time: {step.stats.processing_time:.2f} seconds")
                
                # Update progress
                completed_steps += 1
                self._update_progress(completed_steps, total_steps)
                
            except PipelineError as e:
                # Handle pipeline-specific errors
                success = False
                self.logger.error(f"Pipeline error in step {step.name}: {e}")
                
                # Check if we should continue processing
                if not ErrorHandler.should_continue_processing(e):
                    self.logger.error("Fatal error encountered. Stopping pipeline execution.")
                    break
                    
            except Exception as e:
                # Handle unexpected errors
                success = False
                self.logger.error(f"Unexpected error in step {step.name}: {e}", exc_info=True)
                
                # Continue with next step
                self.logger.info("Continuing with next step...")

        # After all pipeline steps are completed, the main process uniformly cleans up all intermediate directories
        if self.config.cleanup_intermediate_files:
            try:
                for dir_path in [self.config.decompressed_dir, self.config.nifti_dir, self.config.registered_dir, self.config.skull_stripped_dir]:
                    if dir_path.exists():
                        shutil.rmtree(dir_path)
                        self.logger.info(f"[Cleanup] Removed intermediate directory: {dir_path}")
            except Exception as cleanup_err:
                self.logger.warning(f"[Cleanup] Failed to remove intermediate directory after pipeline: {cleanup_err}")
        
        # Calculate total processing time
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        # Generate final report
        self._generate_final_report(total_time)
        
        return success
    
    def _update_progress(self, completed: int, total: int) -> None:
        """
        Update progress tracking
        
        Args:
            completed: Number of completed steps
            total: Total number of steps
        """
        progress_pct = (completed / total) * 100 if total > 0 else 0
        self.logger.info(f"Pipeline progress: {completed}/{total} steps ({progress_pct:.1f}%)")
        
        # Calculate elapsed time and estimate remaining time
        elapsed_time = time.time() - self.start_time
        if completed > 0:
            estimated_total_time = (elapsed_time / completed) * total
            remaining_time = estimated_total_time - elapsed_time
            self.logger.info(f"Elapsed time: {elapsed_time:.1f}s, Estimated remaining: {remaining_time:.1f}s")
    
    def _generate_final_report(self, total_time: float) -> None:
        """
        Generate a final report of pipeline execution
        
        Args:
            total_time: Total processing time in seconds
        """
        self.logger.info("=" * 50)
        self.logger.info("Pipeline Execution Summary")
        self.logger.info("=" * 50)
        self.logger.info(f"Total processing time: {total_time:.2f} seconds")
        
        # Calculate overall statistics
        total_cases = 0
        successful_cases = 0
        failed_cases = 0
        
        # Report on each step
        self.logger.info("\nStep-by-Step Summary:")
        for step in self.steps:
            if not step.enabled:
                self.logger.info(f"- {step.name}: SKIPPED")
                continue
                
            success_rate = 0
            if step.stats.total_cases > 0:
                success_rate = (step.stats.successful_cases / step.stats.total_cases) * 100
                
            self.logger.info(f"- {step.name}:")
            self.logger.info(f"  Success rate: {step.stats.successful_cases}/{step.stats.total_cases} cases ({success_rate:.1f}%)")
            self.logger.info(f"  Processing time: {step.stats.processing_time:.2f} seconds")
            
            # Accumulate statistics
            total_cases += step.stats.total_cases
            successful_cases += step.stats.successful_cases
            failed_cases += step.stats.failed_cases
        
        # Overall statistics
        self.logger.info("\nOverall Statistics:")
        self.logger.info(f"Total cases processed: {total_cases}")
        self.logger.info(f"Successful cases: {successful_cases}")
        self.logger.info(f"Failed cases: {failed_cases}")
        
        if total_cases > 0:
            overall_success_rate = (successful_cases / total_cases) * 100
            self.logger.info(f"Overall success rate: {overall_success_rate:.1f}%")
        
        self.logger.info("=" * 50)
    
    def get_step_by_name(self, name: str) -> Optional[PipelineStep]:
        """
        Get a step by its name
        
        Args:
            name: The name of the step to find
            
        Returns:
            The step if found, None otherwise
        """
        for step in self.steps:
            if step.name == name:
                return step
        return None
    
    def enable_step(self, name: str) -> bool:
        """
        Enable a step by its name
        
        Args:
            name: The name of the step to enable
            
        Returns:
            True if the step was found and enabled, False otherwise
        """
        step = self.get_step_by_name(name)
        if step:
            step.enabled = True
            self.logger.info(f"Enabled step: {name}")
            return True
        return False
    
    def disable_step(self, name: str) -> bool:
        """
        Disable a step by its name
        
        Args:
            name: The name of the step to disable
            
        Returns:
            True if the step was found and disabled, False otherwise
        """
        step = self.get_step_by_name(name)
        if step:
            step.enabled = False
            self.logger.info(f"Disabled step: {name}")
            return True
        return False