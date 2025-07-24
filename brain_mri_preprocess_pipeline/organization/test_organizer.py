#!/usr/bin/env python3
"""
Test script for the FileOrganizer class.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path to allow importing from parent modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brain_mri_preprocess_pipeline.organization.file_organizer import FileOrganizer
from brain_mri_preprocess_pipeline.encryption.case_number_encryptor import CaseNumberEncryptor
from brain_mri_preprocess_pipeline.utils.logging_utils import setup_logging

def test_file_organizer():
    """Test the FileOrganizer class"""
    # Set up logging
    log_dir = Path("logs")
    logger = setup_logging(log_dir, "test_organizer")
    
    # Set up test directories
    input_dir = Path("test_input")
    output_dir = Path("test_output")
    
    # Create test directories if they don't exist
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Create a test case directory
    case_dir = input_dir / "10005023486 MONROE DARLENE E"
    case_dir.mkdir(exist_ok=True)
    
    # Create test files with modality names in them
    test_files = {
        "DWI": case_dir / "DWI_sequence.nii.gz",
        "ADC": case_dir / "ADC_sequence.nii.gz",
        "FLAIR": case_dir / "FLAIR_sequence.nii.gz"
    }
    
    # Create empty test files
    for modality, file_path in test_files.items():
        with open(file_path, 'w') as f:
            f.write(f"Test {modality} file")
    
    # Set up encryptor
    encryption_key = "test_key"
    mapping_file = output_dir / "case_mapping.json"
    encryptor = CaseNumberEncryptor(encryption_key, mapping_file, logger)
    
    # Create FileOrganizer
    organizer = FileOrganizer(input_dir, output_dir, encryptor, logger)
    
    # Test organize_case
    success = organizer.organize_case(case_dir)
    logger.info(f"organize_case result: {success}")
    
    # Test process_directory
    total, successful = organizer.process_directory()
    logger.info(f"process_directory result: {total} total, {successful} successful")
    
    # Test preserve_original_structure
    organizer.preserve_original_structure(input_dir, output_dir)
    
    # List organized files
    logger.info("Organized files:")
    for file_path in output_dir.glob("*.nii.gz"):
        logger.info(f"  {file_path}")
    
    logger.info("Test completed")

if __name__ == "__main__":
    test_file_organizer()