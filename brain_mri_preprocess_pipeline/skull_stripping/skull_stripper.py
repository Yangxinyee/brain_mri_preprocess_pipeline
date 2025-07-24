#!/usr/bin/env python3
"""
Skull stripping module for the medical image processing pipeline.
Handles skull stripping of brain MRI images using HD-BET tool.

This module implements requirement 4.1, 4.2, and 4.3:
- Perform skull stripping using HD-BET tool
- Use the HD-BET tool installed in the nnunet-gpu conda environment
- Handle failures gracefully and continue processing
"""

import os
import time
import subprocess
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Any

from ..utils.models import ProcessingStats, PipelineError
from ..utils.logging_utils import LogManager
from ..utils.environment import EnvironmentManager

class SkullStrippingProcessor:
    """Handles skull stripping of brain MRI images using HD-BET tool"""
    
    def __init__(self, input_dir: Path, output_dir: Path, log_manager: LogManager):
        """
        Initialize the skull stripping processor
        
        Args:
            input_dir: Directory containing registered NIfTI files
            output_dir: Directory where skull-stripped files will be stored
            log_manager: Logging manager
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.logger = log_manager.setup_step_logger("skull_stripping")
        self.env_manager = EnvironmentManager(self.logger)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Skull Stripping Processor initialized")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def check_hdbet_installation(self) -> bool:
        """
        Check if HD-BET is properly installed in the nnunet-gpu environment
        
        Returns:
            True if HD-BET is installed, False otherwise
        """
        return self.env_manager.check_tool_availability("hd-bet")
    
    def perform_skull_stripping(self, input_file: Path, output_file: Path) -> bool:
        """
        Perform skull stripping on a single file using HD-BET
        
        Args:
            input_file: Path to input NIfTI file
            output_file: Path where skull-stripped file will be stored
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Extract case number from filename
            # ISLES2022_55349912_0000.nii.gz -> 55349912
            case_number = input_file.stem.split('_')[1] if '_' in input_file.stem else input_file.stem
            
            self.logger.info(f"Performing skull stripping for case {case_number}")
            
            # HD-BET command (GPU, default)
            cmd = [
                'hd-bet',
                '-i', str(input_file),
                '-o', str(output_file)
            ]
            
            self.logger.info(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Successfully skull-stripped case {case_number}")
                return True
            else:
                self.logger.error(f"Error skull-stripping case {case_number}: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Exception during skull stripping: {e}")
            return False
    
    def verify_skull_stripping(self, processed_image: Path) -> bool:
        """
        Verify the quality of skull stripping
        
        Args:
            processed_image: Path to skull-stripped image
            
        Returns:
            True if quality is acceptable, False otherwise
        """
        # In a real implementation, this would perform quality checks
        # For now, just check if the file exists and is not empty
        try:
            if not processed_image.exists():
                self.logger.error(f"Skull-stripped image does not exist: {processed_image}")
                return False
                
            if processed_image.stat().st_size == 0:
                self.logger.error(f"Skull-stripped image is empty: {processed_image}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying skull stripping quality: {e}")
            return False
    
    def process_case(self, case_id: str) -> bool:
        """
        Process a single case
        
        Args:
            case_id: Case identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Find all modality files for this case
            # We need to process all three modalities: DWI (0000), ADC (0001), and FLAIR (0002)
            modality_suffixes = ["0000", "0001", "0002_registered"]
            
            for suffix in modality_suffixes:
                # Find the input file
                pattern = f"*{case_id}*_{suffix}.nii.gz"
                input_files = list(self.input_dir.rglob(pattern))
                
                if not input_files:
                    self.logger.warning(f"No files found for case {case_id} with suffix {suffix}")
                    continue
                
                input_file = input_files[0]
                
                # Define output file path - replace _registered with _skull_stripped if present
                output_suffix = suffix.replace("_registered", "")
                output_file = self.output_dir / f"ISLES2022_{case_id}_{output_suffix}_skull_stripped.nii.gz"
                
                # Perform skull stripping
                if not self.perform_skull_stripping(input_file, output_file):
                    self.logger.error(f"Skull stripping failed for case {case_id}, modality {suffix}")
                    return False
                
                # Verify skull stripping quality
                if not self.verify_skull_stripping(output_file):
                    self.logger.error(f"Skull stripping quality verification failed for case {case_id}, modality {suffix}")
                    return False
            
            return True
                
        except Exception as e:
            self.logger.error(f"Error processing case {case_id}: {e}")
            return False
    
    def find_case_ids(self) -> List[str]:
        """
        Find all case IDs in the input directory
        
        Returns:
            List of case IDs
        """
        case_ids = set()
        
        # Find all files
        all_files = list(self.input_dir.rglob("*.nii.gz"))
        
        for file_path in all_files:
            # Extract case number from filename
            # ISLES2022_55349912_0000.nii.gz -> 55349912
            filename = file_path.name
            parts = filename.split('_')
            if len(parts) >= 3:
                case_id = parts[1]
                case_ids.add(case_id)
        
        return list(case_ids)
    
    def process_directory(self, parallel: bool = False, max_workers: int = 4) -> ProcessingStats:
        """
        Process all cases in the input directory
        
        Args:
            parallel: Whether to process cases in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            Processing statistics
        """
        start_time = time.time()
        
        # Check HD-BET installation
        if not self.check_hdbet_installation():
            error_msg = "HD-BET is not installed or not in PATH"
            self.logger.error(error_msg)
            raise PipelineError("skull_stripping", "N/A", error_msg, False)
        
        # Find all case IDs
        case_ids = self.find_case_ids()
        total_cases = len(case_ids)
        
        self.logger.info(f"Found {total_cases} cases to process")
        
        # Initialize statistics
        stats = ProcessingStats(
            total_cases=total_cases,
            successful_cases=0,
            failed_cases=0,
            processing_time=0.0,
            error_details=[]
        )
        
        if total_cases == 0:
            self.logger.warning(f"No cases found in {self.input_dir}")
            stats.processing_time = time.time() - start_time
            return stats
        
        # Process cases
        if parallel and total_cases > 1:
            self.logger.info(f"Processing cases in parallel with {max_workers} workers")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_case = {
                    executor.submit(self.process_case, case_id): case_id 
                    for case_id in case_ids
                }
                
                # Process results as they complete
                for future in as_completed(future_to_case):
                    case_id = future_to_case[future]
                    try:
                        success = future.result()
                        if success:
                            stats.successful_cases += 1
                        else:
                            stats.failed_cases += 1
                            stats.error_details.append(f"Failed to process case: {case_id}")
                    except Exception as e:
                        stats.failed_cases += 1
                        error_msg = f"Exception processing case {case_id}: {e}"
                        stats.error_details.append(error_msg)
                        self.logger.error(error_msg)
        else:
            self.logger.info("Processing cases sequentially")
            for i, case_id in enumerate(case_ids, 1):
                self.logger.info(f"Processing case {i}/{total_cases}: {case_id}")
                
                try:
                    if self.process_case(case_id):
                        stats.successful_cases += 1
                    else:
                        stats.failed_cases += 1
                        stats.error_details.append(f"Failed to process case: {case_id}")
                except Exception as e:
                    stats.failed_cases += 1
                    error_msg = f"Exception processing case {case_id}: {e}"
                    stats.error_details.append(error_msg)
                    self.logger.error(error_msg)
        
        # Calculate processing time
        stats.processing_time = time.time() - start_time
        
        # Log summary
        self.logger.info(f"Skull stripping summary:")
        self.logger.info(f"  Total cases: {stats.total_cases}")
        self.logger.info(f"  Successful: {stats.successful_cases}")
        self.logger.info(f"  Failed: {stats.failed_cases}")
        self.logger.info(f"  Processing time: {stats.processing_time:.2f} seconds")
        
        if stats.error_details:
            self.logger.info("Error details:")
            for error in stats.error_details[:10]:  # Limit to first 10 errors
                self.logger.info(f"  - {error}")
            
            if len(stats.error_details) > 10:
                self.logger.info(f"  ... and {len(stats.error_details) - 10} more errors")
        
        return stats