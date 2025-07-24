#!/usr/bin/env python3
"""
Logging utilities for the medical image processing pipeline.
Provides centralized logging configuration and management.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

class LogManager:
    """Centralized logging system for all pipeline operations"""
    
    def __init__(self, log_dir: Path):
        """Initialize the log manager"""
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.loggers: Dict[str, logging.Logger] = {}
        
        # Set up the root logger
        self.setup_root_logger()
    
    def setup_root_logger(self):
        """Set up the root logger with console output"""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Create console handler if not already present
        has_console_handler = any(isinstance(h, logging.StreamHandler) and 
                                h.stream == sys.stdout for h in root_logger.handlers)
        
        if not has_console_handler:
            # Create console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            
            # Add handler to root logger
            root_logger.addHandler(console_handler)
    
    def setup_step_logger(self, step_name: str) -> logging.Logger:
        """Set up a logger for a specific processing step"""
        # Check if logger already exists
        if step_name in self.loggers:
            return self.loggers[step_name]
        
        # Create a logger for this step
        logger = logging.getLogger(step_name)
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers to avoid duplicate logs
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create a file handler for this step
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{step_name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        # Store logger for future reference
        self.loggers[step_name] = logger
        
        return logger
    
    def log_processing_stats(self, step_name: str, stats: dict):
        """Log processing statistics for a step"""
        logger = self.get_or_create_logger(step_name)
        
        logger.info(f"Processing summary for {step_name}:")
        logger.info(f"  Total cases: {stats.get('total_cases', 0)}")
        logger.info(f"  Successful cases: {stats.get('successful_cases', 0)}")
        logger.info(f"  Failed cases: {stats.get('failed_cases', 0)}")
        logger.info(f"  Processing time: {stats.get('processing_time', 0):.2f} seconds")
        
        error_details = stats.get('error_details', [])
        if error_details:
            logger.info("Error details:")
            for error in error_details:
                logger.info(f"  - {error}")
    
    def get_or_create_logger(self, step_name: str) -> logging.Logger:
        """Get an existing logger or create a new one if it doesn't exist"""
        if step_name not in self.loggers:
            return self.setup_step_logger(step_name)
        return self.loggers[step_name]
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of all processing steps"""
        summary = "Pipeline Processing Summary\n"
        summary += "=" * 30 + "\n"
        
        # In a real implementation, this would parse log files and extract statistics
        for step_name in self.loggers:
            summary += f"Step: {step_name}\n"
            # Here we would add statistics from each step
        
        return summary


def setup_logging(log_dir: Path, step_name: Optional[str] = None) -> logging.Logger:
    """
    Set up logging for a script or module.
    
    Args:
        log_dir: Directory where log files will be stored
        step_name: Name of the processing step (optional)
        
    Returns:
        Logger instance
    """
    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up basic logging configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{step_name or 'general'}_{timestamp}.log"
    log_path = log_dir / log_filename
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Get logger
    logger = logging.getLogger(step_name or 'root')
    
    logger.info(f"Logging initialized. Log file: {log_path}")
    return logger