#!/usr/bin/env python3
"""
Configuration utilities for the medical image processing pipeline.
Handles pipeline configuration and validation.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

# Default values
DEFAULT_LOG_DIR = "logs"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_CONDA_ENV = "nnunet-gpu"

@dataclass
class PipelineConfig:
    """Configuration for the pipeline"""
    
    # Required parameters
    input_directory: Path
    
    # Optional parameters with defaults
    output_directory: Path = field(default_factory=lambda: Path(DEFAULT_OUTPUT_DIR))
    log_directory: Path = field(default_factory=lambda: Path(DEFAULT_LOG_DIR))
    conda_environment: str = DEFAULT_CONDA_ENV
    encryption_key_file: Optional[Path] = None
    parallel_workers: int = 1
    cleanup_intermediate_files: bool = False
    
    # Derived paths
    decompressed_dir: Path = field(init=False)
    nifti_dir: Path = field(init=False)
    registered_dir: Path = field(init=False)
    skull_stripped_dir: Path = field(init=False)
    final_dir: Path = field(init=False)
    
    def __post_init__(self):
        """Initialize derived paths and create directories"""
        # Convert string paths to Path objects if needed
        if isinstance(self.input_directory, str):
            self.input_directory = Path(self.input_directory)
        
        if isinstance(self.output_directory, str):
            self.output_directory = Path(self.output_directory)
            
        if isinstance(self.log_directory, str):
            self.log_directory = Path(self.log_directory)
            
        if isinstance(self.encryption_key_file, str):
            self.encryption_key_file = Path(self.encryption_key_file)
        
        # Create directories if they don't exist
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Set up derived paths
        self.decompressed_dir = self.output_directory / "decompressed"
        self.nifti_dir = self.output_directory / "nifti"
        self.registered_dir = self.output_directory / "registered"
        self.skull_stripped_dir = self.output_directory / "skull_stripped"
        self.final_dir = self.output_directory / "Dataset002_ISLES2022_all/imagesTs"
        
        # Create derived directories
        self.decompressed_dir.mkdir(parents=True, exist_ok=True)
        self.nifti_dir.mkdir(parents=True, exist_ok=True)
        self.registered_dir.mkdir(parents=True, exist_ok=True)
        self.skull_stripped_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)
    
    def validate(self, logger: Optional[logging.Logger] = None) -> bool:
        """Validate the configuration"""
        logger = logger or logging.getLogger(__name__)
        
        # Check if input directory exists
        if not self.input_directory.exists():
            logger.error(f"Input directory does not exist: {self.input_directory}")
            return False
        
        # Check if we have write permissions to output directory
        try:
            test_file = self.output_directory / ".write_test"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            logger.error(f"Cannot write to output directory: {e}")
            return False
        
        # Check if encryption key file exists if specified
        if self.encryption_key_file and not self.encryption_key_file.exists():
            logger.error(f"Encryption key file does not exist: {self.encryption_key_file}")
            return False
        
        # Check if parallel workers is valid
        if self.parallel_workers < 1:
            logger.error(f"Invalid number of parallel workers: {self.parallel_workers}")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "input_directory": str(self.input_directory),
            "output_directory": str(self.output_directory),
            "log_directory": str(self.log_directory),
            "conda_environment": self.conda_environment,
            "encryption_key_file": str(self.encryption_key_file) if self.encryption_key_file else None,
            "parallel_workers": self.parallel_workers,
            "cleanup_intermediate_files": self.cleanup_intermediate_files,
            "decompressed_dir": str(self.decompressed_dir),
            "nifti_dir": str(self.nifti_dir),
            "registered_dir": str(self.registered_dir),
            "skull_stripped_dir": str(self.skull_stripped_dir),
            "final_dir": str(self.final_dir)
        }