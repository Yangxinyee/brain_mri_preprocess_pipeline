#!/usr/bin/env python3
"""
Medical Image Processing Pipeline for Stroke MRI Data

This script orchestrates the complete processing pipeline for stroke MRI data:
1. DICOM decompression
2. DICOM to NIfTI conversion
3. FLAIR to DWI registration
4. Skull stripping using HD-BET
5. Case number encryption
6. File organization into standardized structure

Author: Medical Imaging Team
Date: 2025-07-20
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime

# Import utility modules
from brain_mri_preprocess_pipeline.utils.logging_utils import LogManager
from brain_mri_preprocess_pipeline.utils.environment import EnvironmentManager
from brain_mri_preprocess_pipeline.utils.config import PipelineConfig
from brain_mri_preprocess_pipeline.utils.models import ProcessingStats, PipelineError
from brain_mri_preprocess_pipeline.utils.pipeline_orchestrator import PipelineOrchestrator, PipelineStep

# Import processing modules
from brain_mri_preprocess_pipeline.decompression.dicom_decompressor import DicomDecompressor
from brain_mri_preprocess_pipeline.conversion.dicom_to_nifti import DicomToNiftiConverter
from brain_mri_preprocess_pipeline.registration.registration_engine import ImageRegistrationEngine
from brain_mri_preprocess_pipeline.skull_stripping.skull_stripper import SkullStrippingProcessor
from brain_mri_preprocess_pipeline.encryption.case_number_encryptor import CaseNumberEncryptor
from brain_mri_preprocess_pipeline.organization.file_organizer import FileOrganizer

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Medical Image Processing Pipeline for Stroke MRI Data")
    
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input directory containing the DICOM files")
    parser.add_argument("--output", "-o", type=str, default="output",
                        help="Output directory for processed files (default: output)")
    parser.add_argument("--logs", "-l", type=str, default="logs",
                        help="Directory for log files (default: logs)")
    parser.add_argument("--key-file", "-k", type=str,
                        help="File containing the encryption key for case numbers")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--cleanup", "-c", action="store_true",
                        help="Clean up intermediate files after processing")
    parser.add_argument("--skip-steps", type=str,
                        help="Comma-separated list of steps to skip (e.g., 'decompression,conversion')")
    parser.add_argument("--only-steps", type=str,
                        help="Comma-separated list of steps to run (e.g., 'registration,skull_stripping')")
    
    return parser.parse_args()

def main():
    """Main function"""
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Create configuration
    config = PipelineConfig(
        input_directory=Path(args.input),
        output_directory=Path(args.output),
        log_directory=Path(args.logs),
        encryption_key_file=Path(args.key_file) if args.key_file else None,
        parallel_workers=args.workers,
        cleanup_intermediate_files=args.cleanup
    )
    
    # Set up logging
    log_manager = LogManager(config.log_directory)
    main_logger = log_manager.setup_step_logger("main")
    
    main_logger.info("Starting Medical Image Processing Pipeline")
    main_logger.info(f"Input directory: {config.input_directory}")
    main_logger.info(f"Output directory: {config.output_directory}")
    
    # Validate configuration
    if not config.validate(main_logger):
        main_logger.error("Configuration validation failed. Exiting.")
        return 1
    
    # Verify environment
    env_manager = EnvironmentManager(main_logger)
    if not env_manager.verify_environment():
        main_logger.error("Required conda environment 'nnunet-gpu' is not activated.")
        main_logger.error("Please activate the environment with 'conda activate nnunet-gpu' and try again.")
        return 1
    
    # Check required tools
    required_tools = ["dcm2niix", "antsRegistration", "hd-bet"]
    if not env_manager.check_required_tools(required_tools):
        main_logger.error("One or more required tools are missing. Please install them and try again.")
        return 1
    
    main_logger.info("Environment verification completed successfully.")
    
    try:
        # Initialize the pipeline orchestrator
        orchestrator = PipelineOrchestrator(config, log_manager)
        
        # Initialize processing modules
        decompressor = DicomDecompressor(
            input_dir=config.input_directory,
            output_dir=config.decompressed_dir,
            log_manager=log_manager
        )
        
        converter = DicomToNiftiConverter(
            input_dir=config.decompressed_dir,
            output_dir=config.nifti_dir,
            log_manager=log_manager
        )
        
        registration_engine = ImageRegistrationEngine(
            input_dir=config.nifti_dir,
            output_dir=config.registered_dir,
            log_manager=log_manager
        )
        
        skull_stripper = SkullStrippingProcessor(
            input_dir=config.nifti_dir,
            output_dir=config.skull_stripped_dir,
            log_manager=log_manager
        )
        
        encryptor = CaseNumberEncryptor(
            encryption_key_file=config.encryption_key_file,
            log_manager=log_manager
        )
        
        file_organizer = FileOrganizer(
            input_dir=config.skull_stripped_dir,
            output_dir=config.final_dir,
            encryptor=encryptor,
            logger=log_manager.setup_step_logger("organization")
        )
        
        # Add pipeline steps
        orchestrator.add_step(PipelineStep(
            name="decompression",
            description="DICOM Decompression",
            processor=decompressor,
            method_name="process_directory"
        ))
        
        orchestrator.add_step(PipelineStep(
            name="conversion",
            description="DICOM to NIfTI Conversion",
            processor=converter,
            method_name="process_directory"
        ))
        
        orchestrator.add_step(PipelineStep(
            name="registration",
            description="FLAIR to DWI Registration",
            processor=registration_engine,
            method_name="process_directory"
        ))
        
        orchestrator.add_step(PipelineStep(
            name="skull_stripping",
            description="Skull Stripping",
            processor=skull_stripper,
            method_name="process_directory"
        ))
        
        orchestrator.add_step(PipelineStep(
            name="encryption",
            description="Case Number Encryption",
            processor=encryptor,
            method_name="process_directory"
        ))
        
        orchestrator.add_step(PipelineStep(
            name="organization",
            description="File Organization",
            processor=file_organizer,
            method_name="process_directory"
        ))
        
        # Handle step selection
        if args.skip_steps:
            skip_steps = [step.strip() for step in args.skip_steps.split(',')]
            for step_name in skip_steps:
                orchestrator.disable_step(step_name)
                main_logger.info(f"Skipping step: {step_name}")
        
        if args.only_steps:
            only_steps = [step.strip() for step in args.only_steps.split(',')]
            # First disable all steps
            for step in orchestrator.steps:
                step.enabled = False
            # Then enable only the specified steps
            for step_name in only_steps:
                orchestrator.enable_step(step_name)
                main_logger.info(f"Only running step: {step_name}")
        
        # Execute the pipeline
        success = orchestrator.execute_pipeline(parallel=(config.parallel_workers > 1))
        
        if not success:
            main_logger.warning("Pipeline completed with errors. Check logs for details.")
            return 1
        
    except Exception as e:
        main_logger.error(f"Unhandled exception: {e}", exc_info=True)
        return 1
    
    main_logger.info("Pipeline execution completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())