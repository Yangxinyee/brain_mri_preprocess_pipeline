#!/usr/bin/env python3
"""
Image registration module for the medical image processing pipeline.
Handles registration of FLAIR images to DWI space using ANTs.

Based on the reference implementation in resample_flair_to_dwi.py
"""

import os
import time
import subprocess
import logging
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Any
import shutil

from ..utils.models import ProcessingStats, PipelineError
from ..utils.logging_utils import LogManager

class ImageRegistrationEngine:
    """Handles registration of FLAIR images to DWI space using ANTs"""
    
    def __init__(self, input_dir: Path, output_dir: Path, log_manager: LogManager):
        """
        Initialize the image registration engine
        
        Args:
            input_dir: Directory containing NIfTI files
            output_dir: Directory where registered files will be stored
            log_manager: Logging manager
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.logger = log_manager.setup_step_logger("image_registration")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Image Registration Engine initialized")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def check_ants_installation(self) -> bool:
        """
        Check if ANTs is properly installed
        
        Returns:
            True if ANTs is installed, False otherwise
        """
        try:
            result = subprocess.run(
                ['which', 'antsRegistration'], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                ants_path = result.stdout.strip()
                self.logger.info(f"ANTs found at: {ants_path}")
                return True
            else:
                self.logger.error("ANTs not found. Please install ANTs first.")
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking ANTs installation: {e}")
            return False
    
    def register_flair_to_dwi(self, dwi_file: Path, flair_file: Path, output_file: Path) -> bool:
        """
        Register FLAIR image to DWI space using ANTs
        
        Args:
            dwi_file: Path to DWI NIfTI file
            flair_file: Path to FLAIR NIfTI file
            output_file: Path where registered FLAIR file will be stored
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Extract case number from filename
            # ISLES2022_55349912_0000.nii.gz -> 55349912
            case_number = dwi_file.stem.split('_')[1] if '_' in dwi_file.stem else dwi_file.stem
            
            self.logger.info(f"Registering FLAIR to DWI for case {case_number}")
            
            # ANTs registration command
            transform_prefix = str(output_file.parent / f"{case_number}_transform")
            
            cmd = [
                'antsRegistration',
                '--dimensionality', '3',
                '--float', '0',
                '--output', f'[{transform_prefix},{output_file}]',
                '--interpolation', 'Linear',
                '--winsorize-image-intensities', '[0.005,0.995]',
                '--use-histogram-matching', '0',
                '--initial-moving-transform', f'[{dwi_file},{flair_file},1]',
                '--transform', 'Rigid[0.1]',
                '--metric', f'MI[{dwi_file},{flair_file},1,32,Regular,0.25]',
                '--convergence', '[500x250x100,1e-6,10]',
                '--shrink-factors', '4x2x1',
                '--smoothing-sigmas', '2x1x0vox',
                '--transform', 'Affine[0.1]',
                '--metric', f'MI[{dwi_file},{flair_file},1,32,Regular,0.25]',
                '--convergence', '[500x250x100,1e-6,10]',
                '--shrink-factors', '4x2x1',
                '--smoothing-sigmas', '2x1x0vox'
            ]
            
            self.logger.info(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Successfully registered case {case_number}")
                
                # Clean up transform files
                for ext in ['0GenericAffine.mat', '1Warp.nii.gz', '1InverseWarp.nii.gz']:
                    transform_file = f"{transform_prefix}{ext}"
                    if os.path.exists(transform_file):
                        os.remove(transform_file)
                        
                return True
            else:
                self.logger.error(f"Error registering case {case_number}: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Exception during registration: {e}")
            return False
    
    def verify_registration_quality(self, registered_image: Path) -> bool:
        """
        Verify the quality of registration
        
        Args:
            registered_image: Path to registered image
            
        Returns:
            True if quality is acceptable, False otherwise
        """
        # In a real implementation, this would perform quality checks
        # For now, just check if the file exists and is not empty
        try:
            if not registered_image.exists():
                self.logger.error(f"Registered image does not exist: {registered_image}")
                return False
                
            if registered_image.stat().st_size == 0:
                self.logger.error(f"Registered image is empty: {registered_image}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying registration quality: {e}")
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
            # Find DWI, FLAIR, and ADC files
            dwi_pattern = f"*{case_id}*_0000.nii.gz"
            flair_pattern = f"*{case_id}*_0002.nii.gz"
            adc_pattern = f"*{case_id}*_0001.nii.gz"
            
            dwi_files = list(self.input_dir.rglob(dwi_pattern))
            flair_files = list(self.input_dir.rglob(flair_pattern))
            adc_files = list(self.input_dir.rglob(adc_pattern))
            
            if not dwi_files:
                self.logger.error(f"DWI file not found for case {case_id}")
                return False
                
            if not flair_files:
                self.logger.error(f"FLAIR file not found for case {case_id}")
                return False
                
            dwi_file = dwi_files[0]
            flair_file = flair_files[0]
            
            # Load and check all modalities
            import nibabel as nib
            import numpy as np
            
            dwi_img = nib.load(str(dwi_file))
            flair_img = nib.load(str(flair_file))
            
            dwi_is_zero = np.all(dwi_img.get_fdata() == 0)
            flair_is_zero = np.all(flair_img.get_fdata() == 0)
            
            # Check ADC if it exists
            adc_is_zero = True  # Default to True if ADC doesn't exist
            if adc_files:
                adc_file = adc_files[0]
                adc_img = nib.load(str(adc_file))
                adc_is_zero = np.all(adc_img.get_fdata() == 0)
            else:
                adc_file = None
            
            # Check if all modalities are zero
            if dwi_is_zero and flair_is_zero and adc_is_zero:
                self.logger.warning(f"All modalities are zero for case {case_id}, skipping case.")
                # Move case to skipped_cases
                skipped_dir = self.input_dir.parent.parent / "skipped_cases"
                skipped_dir.mkdir(parents=True, exist_ok=True)
                case_dir = dwi_file.parent
                shutil.move(str(case_dir), str(skipped_dir / case_dir.name))
                return False
            
            # Determine registration strategy
            skip_registration = False
            registration_template = None
            
            if flair_is_zero:
                # Scenario 1: Normal DWI, Zero FLAIR or Scenario 3: Zero DWI, Zero FLAIR, Normal ADC
                self.logger.info(f"FLAIR is zero for case {case_id}, skipping registration - treating as already registered.")
                skip_registration = True
            elif dwi_is_zero and (not adc_file or adc_is_zero):
                # Scenario 4: Zero DWI, Normal FLAIR, Zero ADC (or ADC does not exist)
                self.logger.info(f"No valid registration template (DWI and ADC are zero) for case {case_id}, skipping registration - treating FLAIR as already registered.")
                skip_registration = True
            elif dwi_is_zero and not adc_is_zero:
                # Scenario 2: Zero DWI, Normal FLAIR, Normal ADC - use ADC as registration template
                self.logger.warning(f"DWI is zero for case {case_id}, using ADC for registration.")
                registration_template = adc_file
            else:
                # Normal case: Normal DWI, Normal FLAIR
                registration_template = dwi_file
            
            if skip_registration:
                # Skip registration but continue processing
                # Create the "registered" file by copying the original FLAIR
                case_dir = flair_file.parent
                target_registered = case_dir / f"ISLES2022_{case_id}_0002_registered.nii.gz"
                
                # Copy original FLAIR to registered name
                shutil.copy2(str(flair_file), str(target_registered))
                
                # Delete original FLAIR file
                flair_file.unlink()
                
                self.logger.info(f"Registration skipped for case {case_id}, FLAIR treated as already registered.")
                return True
            
            # Proceed with actual registration
            output_file = self.output_dir / f"ISLES2022_{case_id}_0002_registered.nii.gz"
            
            # Register FLAIR to the selected template
            if self.register_flair_to_dwi(registration_template, flair_file, output_file):
                # Verify registration quality
                if self.verify_registration_quality(output_file):
                    # Move registered file to case directory
                    case_dir_name = registration_template.parent.name
                    nifti_case_dir = self.input_dir / case_dir_name
                    nifti_case_dir.mkdir(parents=True, exist_ok=True)
                    target_registered = nifti_case_dir / f"ISLES2022_{case_id}_0002_registered.nii.gz"
                    shutil.move(str(output_file), str(target_registered))
                    
                    # Delete original FLAIR file
                    orig_flair_path = nifti_case_dir / f"ISLES2022_{case_id}_0002.nii.gz"
                    if orig_flair_path.exists():
                        orig_flair_path.unlink()
                        
                    return True
                else:
                    self.logger.error(f"Registration quality verification failed for case {case_id}")
                    return False
            else:
                self.logger.error(f"Registration failed for case {case_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing case {case_id}: {e}")
            return False
    
    def find_case_ids(self) -> List[str]:
        """
        Find all case IDs in the input directory (recursively)
        Returns:
            List of case IDs
        """
        case_ids = set()
        # Recursively find all *_0000.nii.gz files in subdirectories
        dwi_files = list(self.input_dir.rglob("*_0000.nii.gz"))
        for dwi_file in dwi_files:
            # Extract case number from filename
            # ISLES2022_55349912_0000.nii.gz -> 55349912
            filename = dwi_file.name
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
        
        # Check ANTs installation
        if not self.check_ants_installation():
            error_msg = "ANTs is not installed or not in PATH"
            self.logger.error(error_msg)
            raise PipelineError("image_registration", "N/A", error_msg, False)
        
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
        self.logger.info(f"Registration summary:")
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