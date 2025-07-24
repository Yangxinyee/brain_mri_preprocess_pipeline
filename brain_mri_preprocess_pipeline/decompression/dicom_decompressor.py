#!/usr/bin/env python3
"""
DICOM decompression module for the medical image processing pipeline.
Handles batch decompression of compressed DICOM files.

Based on the reference implementation in batch_decompress_dicom.py
"""

import os
import time
import pydicom
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Any

from ..utils.models import ProcessingStats, PipelineError
from ..utils.logging_utils import LogManager

class DicomDecompressor:
    """Handles batch decompression of compressed DICOM files"""
    
    def __init__(self, input_dir: Path, output_dir: Path, log_manager: LogManager):
        """
        Initialize the DICOM decompressor
        
        Args:
            input_dir: Directory containing compressed DICOM files
            output_dir: Directory where decompressed files will be stored
            log_manager: Logging manager
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.logger = log_manager.setup_step_logger("dicom_decompression")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"DICOM Decompressor initialized")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def decompress_file(self, dcm_path: Path, preserve_original: bool = False) -> bool:
        """
        Decompress a single DICOM file
        
        Args:
            dcm_path: Path to the DICOM file
            preserve_original: Whether to preserve the original file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine output path
            rel_path = dcm_path.relative_to(self.input_dir)
            output_path = self.output_dir / rel_path
            
            # Create parent directories if they don't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Read DICOM file
            ds = pydicom.dcmread(dcm_path, force=True)
            
            # Check if file is compressed
            if hasattr(ds, 'file_meta') and hasattr(ds.file_meta, 'TransferSyntaxUID'):
                if ds.file_meta.TransferSyntaxUID.is_compressed:
                    self.logger.info(f"Decompressing: {dcm_path}")
                    ds.decompress()  # Use automatic plugin selection
                    ds.save_as(output_path)
                    self.logger.info(f"Successfully decompressed: {dcm_path} -> {output_path}")
                    return True
                else:
                    # Copy uncompressed file to output directory
                    self.logger.debug(f"Already uncompressed: {dcm_path}")
                    ds.save_as(output_path)
                    return True
            else:
                self.logger.warning(f"File does not have TransferSyntaxUID: {dcm_path}")
                # Copy file anyway
                ds.save_as(output_path)
                return True
                
        except Exception as e:
            self.logger.error(f"Error decompressing {dcm_path}: {e}")
            return False
    
    def process_directory(self, directory: Path = None, parallel: bool = False, 
                         max_workers: int = 4) -> ProcessingStats:
        """
        Process all DICOM files in a directory
        
        Args:
            directory: Directory to process (defaults to input_dir)
            parallel: Whether to process files in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            Processing statistics
        """
        start_time = time.time()
        directory = directory or self.input_dir
        
        if not directory.exists():
            error_msg = f"Directory does not exist: {directory}"
            self.logger.error(error_msg)
            raise PipelineError("dicom_decompression", "N/A", error_msg, False)
        
        # Find all .dcm files recursively
        self.logger.info(f"Searching for DICOM files in {directory}")
        dcm_files = list(directory.rglob("*.dcm"))
        total_files = len(dcm_files)
        
        self.logger.info(f"Found {total_files} DICOM files in {directory}")
        
        # Initialize statistics
        stats = ProcessingStats(
            total_cases=total_files,
            successful_cases=0,
            failed_cases=0,
            processing_time=0.0,
            error_details=[]
        )
        
        if total_files == 0:
            self.logger.warning(f"No DICOM files found in {directory}")
            stats.processing_time = time.time() - start_time
            return stats
        
        # Process files
        if parallel and total_files > 10:  # Only use parallel processing for larger datasets
            self.logger.info(f"Processing files in parallel with {max_workers} workers")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self.decompress_file, dcm_file): dcm_file 
                    for dcm_file in dcm_files
                }
                
                # Process results as they complete
                for future in as_completed(future_to_file):
                    dcm_file = future_to_file[future]
                    try:
                        success = future.result()
                        if success:
                            stats.successful_cases += 1
                        else:
                            stats.failed_cases += 1
                            stats.error_details.append(f"Failed to decompress: {dcm_file}")
                    except Exception as e:
                        stats.failed_cases += 1
                        error_msg = f"Exception processing {dcm_file}: {e}"
                        stats.error_details.append(error_msg)
                        self.logger.error(error_msg)
        else:
            self.logger.info("Processing files sequentially")
            for i, dcm_file in enumerate(dcm_files, 1):
                self.logger.info(f"Processing file {i}/{total_files}: {dcm_file}")
                
                if self.decompress_file(dcm_file):
                    stats.successful_cases += 1
                else:
                    stats.failed_cases += 1
                    stats.error_details.append(f"Failed to decompress: {dcm_file}")
        
        # Calculate processing time
        stats.processing_time = time.time() - start_time
        
        # Log summary
        self.logger.info(f"Decompression summary:")
        self.logger.info(f"  Total files: {stats.total_cases}")
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