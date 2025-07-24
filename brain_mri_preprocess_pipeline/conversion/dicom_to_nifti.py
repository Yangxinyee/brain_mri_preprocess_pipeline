#!/usr/bin/env python3
"""
DICOM to NIfTI conversion module for the medical image processing pipeline.
Handles conversion of DICOM files to NIfTI format with modality consistency handling.

Based on the reference implementation in dcm2nii_final.py
"""

import os
import time
import subprocess
import shutil
import logging
import nibabel as nib
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Any, Set
import re

from ..utils.models import ProcessingStats, ModalityStatus, PipelineError, CaseInfo
from ..utils.logging_utils import LogManager
from brain_mri_preprocess_pipeline.utils.models import CHANNEL_MAPPING

# Define a mapping from modality name to channel name for standardization
# CHANNEL_MAPPING = {
#     "ADC": "ADC",
#     "DWI": "DWI",
#     "FLAIR": "FLAIR"
# }
# Keep CHANNEL_MAPPING in models.py as {"DWI": "0000", "ADC": "0001", "FLAIR": "0002"}

class DicomToNiftiConverter:
    """Handles conversion of DICOM files to NIfTI format with modality consistency handling"""
    
    def __init__(self, input_dir: Path, output_dir: Path, log_manager: LogManager):
        """
        Initialize the DICOM to NIfTI converter
        
        Args:
            input_dir: Directory containing decompressed DICOM files
            output_dir: Directory where NIfTI files will be stored
            log_manager: Logging manager
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.logger = log_manager.setup_step_logger("dicom_to_nifti")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Directory for skipped cases
        self.skipped_dir = self.output_dir.parent / "skipped_cases"
        self.skipped_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"DICOM to NIfTI Converter initialized")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Skipped directory: {self.skipped_dir}")
    
    def check_modality_consistency(self, case_dir: Path) -> ModalityStatus:
        """
        Check if FLAIR, DWI, and ADC have the same number of DICOM files
        
        Args:
            case_dir: Directory containing modality subdirectories
            
        Returns:
            ModalityStatus object with consistency information
        """
        modalities = ['FLAIR', 'DWI', 'ADC']
        status = ModalityStatus()
        
        for modality in modalities:
            modality_dir = case_dir / modality
            if modality_dir.exists():
                dcm_files = list(modality_dir.glob("*.dcm"))
                if modality == 'ADC':
                    status.adc_count = len(dcm_files)
                elif modality == 'DWI':
                    status.dwi_count = len(dcm_files)
                elif modality == 'FLAIR':
                    status.flair_count = len(dcm_files)
            else:
                status.missing_modalities.append(modality)
                if modality == 'ADC':
                    status.adc_count = 0
                elif modality == 'DWI':
                    status.dwi_count = 0
                elif modality == 'FLAIR':
                    status.flair_count = 0
        
        # Determine conversion strategy
        status.determine_strategy()
        
        return status
    
    def convert_dicom_to_nifti(self, dicom_dir: Path, output_dir: Path) -> Tuple[bool, List[Path]]:
        """
        Convert DICOM directory to NIfTI using dcm2niix
        
        Args:
            dicom_dir: Directory containing DICOM files
            output_dir: Directory where NIfTI files will be stored
            
        Returns:
            Tuple of (success, list of output files)
        """
        try:
            # Create unique temporary directory to avoid dcm2niix filename conflicts
            import tempfile
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_output_dir = Path(temp_dir)
                
                # Run dcm2niix command to temporary directory
                cmd = [
                    'dcm2niix',
                    '-f', '%f_%s',  # Output filename format
                    '-o', str(temp_output_dir),  # Output directory
                    '-z', 'y',  # Compress output
                    str(dicom_dir)  # Input directory
                ]
                
                self.logger.info(f"Running command: {' '.join(cmd)}")
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    self.logger.info(f"Successfully converted: {dicom_dir}")
                    
                    # Find all .nii.gz files in the temp directory
                    temp_files = list(temp_output_dir.glob("*.nii.gz"))
                    
                    if not temp_files:
                        self.logger.error(f"No NIfTI files generated for: {dicom_dir}")
                        return False, []
                    
                    # Create output directory if it doesn't exist
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Move files from temp directory to final output directory with unique names
                    output_files = []
                    for i, temp_file in enumerate(temp_files):
                        # Create unique filename to avoid conflicts during processing
                        unique_name = f"temp_{dicom_dir.name}_{i}_{temp_file.name}"
                        final_path = output_dir / unique_name
                        
                        import shutil
                        shutil.move(str(temp_file), str(final_path))
                        output_files.append(final_path)
                        self.logger.info(f"Moved {temp_file.name} to {final_path}")
                    
                    return True, output_files
                else:
                    self.logger.error(f"Conversion failed for {dicom_dir}: {result.stderr}")
                    return False, []
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Conversion timeout for {dicom_dir}")
            return False, []
        except Exception as e:
            self.logger.error(f"Error converting {dicom_dir}: {e}")
            return False, []
    
    def create_empty_nifti(self, reference_path: Path, output_path: Path) -> bool:
        """
        Create an empty NIfTI file based on a reference file
        
        Args:
            reference_path: Path to reference NIfTI file
            output_path: Path where empty NIfTI file will be stored
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if reference file exists
            if not reference_path.exists():
                self.logger.error(f"Reference file does not exist: {reference_path}")
                return False
            
            # Load reference NIfTI file
            ref_img = nib.load(reference_path)
            
            # Create empty array with same shape
            empty_data = np.zeros(ref_img.shape, dtype=np.float32)
            
            # Create new NIfTI file with same header
            empty_img = nib.Nifti1Image(empty_data, ref_img.affine, ref_img.header)
            
            # Save empty NIfTI file
            nib.save(empty_img, output_path)
            
            self.logger.info(f"Created empty NIfTI file: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating empty NIfTI file: {e}")
            return False
    
    def safe_rename_nifti(self, orig_file: Path, target_path: Path, modality: str) -> bool:
        """
        Safely rename a NIfTI file, handling potential conflicts
        
        Args:
            orig_file: Original file path
            target_path: Target file path
            modality: Modality name for logging
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Attempting to rename {modality}: {orig_file} -> {target_path}")
            
            # Check if target file already exists
            if target_path.exists():
                self.logger.warning(f"Target file already exists: {target_path}")
                # Remove existing file to avoid conflict
                target_path.unlink()
                self.logger.info(f"Removed existing file: {target_path}")
            
            # Perform the rename
            os.rename(orig_file, target_path)
            self.logger.info(f"Successfully renamed {modality} file")
            return True
            
        except Exception as rename_error:
            self.logger.error(f"Failed to rename {modality} file {orig_file} to {target_path}: {rename_error}")
            return False
    
    def ensure_dwi_is_3d(self, dwi_path: Path):
        """
        If the DWI NIfTI is 4D, extract the volume with the largest b-value and save as 3D, replacing the original file.
        """
        try:
            img = nib.load(str(dwi_path))
            data = img.get_fdata()
            if data.ndim == 4 and data.shape[3] > 1:
                # Try to find bval file
                bval_path = dwi_path.with_suffix('.bval')
                if not bval_path.exists():
                    bval_path = dwi_path.parent / (dwi_path.stem.split('.nii')[0] + '.bval')
                if bval_path.exists():
                    with open(bval_path, 'r') as f:
                        bvals = [float(x) for x in f.read().strip().split()]
                    max_b_idx = int(np.argmax(bvals))
                else:
                    # Fallback: use the last volume
                    self.logger.warning(f"No bval file found for {dwi_path}, using last volume as max b-value.")
                    max_b_idx = data.shape[3] - 1
                # Extract the 3D volume
                data_3d = data[..., max_b_idx]
                img_3d = nib.Nifti1Image(data_3d, img.affine, img.header)
                nib.save(img_3d, str(dwi_path))
                self.logger.info(f"DWI 4D detected, extracted max b-value volume (index {max_b_idx}) and replaced with 3D: {dwi_path}")
            else:
                self.logger.info(f"DWI is already 3D: {dwi_path}")
        except Exception as e:
            self.logger.error(f"Error processing DWI 4D to 3D for {dwi_path}: {e}")

    def process_case(self, case_dir: Path) -> Tuple[bool, Dict[str, Path]]:
        """
        Process a single case directory
        
        Args:
            case_dir: Directory containing modality subdirectories
            
        Returns:
            Tuple of (success, dictionary mapping modality to output file path)
        """
        case_name = case_dir.name
        self.logger.info(f"Processing case: {case_name}")
        # Extract numeric case_id
        m = re.match(r"(\d+)", case_name)
        if m:
            case_id = m.group(1)
        else:
            case_id = case_name
        # All subsequent ISLES2022_... naming will use case_id instead of case_name
        
        # Check modality consistency
        status = self.check_modality_consistency(case_dir)
        
        # Create output directory for this case
        case_output_dir = self.output_dir / case_name
        case_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dictionary to store output file paths
        output_files = {}
        
        if status.is_consistent:
            # All modalities have the same count, convert all normally
            self.logger.info(f"All modalities are consistent for case {case_name}")
            
            for modality in ['ADC', 'DWI', 'FLAIR']:
                modality_dir = case_dir / modality
                if modality_dir.exists():
                    success, files = self.convert_dicom_to_nifti(modality_dir, case_output_dir)
                    if success and files:
                        # --- Rename to standard format ---
                        orig_file = files[0]
                        channel = CHANNEL_MAPPING[modality]
                        std_name = f"ISLES2022_{case_id}_{channel}.nii.gz"
                        std_path = case_output_dir / std_name
                        if self.safe_rename_nifti(orig_file, std_path, modality):
                            output_files[modality] = std_path
                            if modality == 'DWI':
                                self.ensure_dwi_is_3d(std_path)
                        else:
                            return False, {}
                        # --- End ---
                    else:
                        self.logger.error(f"Failed to convert {modality} for case {case_name}")
                        return False, {}
                else:
                    self.logger.error(f"Missing {modality} directory for case {case_name}")
                    return False, {}
            
            return True, output_files
            
        else:
            # Handle inconsistent modalities based on strategy
            self.logger.warning(f"Inconsistent modalities for case {case_name}: {status.strategy}")
            self.logger.warning(f"ADC: {status.adc_count}, DWI: {status.dwi_count}, FLAIR: {status.flair_count}")
            
            if status.strategy == "adc_only":
                # Convert ADC normally, create empty DWI and FLAIR
                adc_dir = case_dir / "ADC"
                if adc_dir.exists() and status.adc_count > 0:
                    success, files = self.convert_dicom_to_nifti(adc_dir, case_output_dir)
                    if success and files:
                        # --- Rename ADC ---
                        orig_file = files[0]
                        channel = CHANNEL_MAPPING["ADC"]
                        std_name = f"ISLES2022_{case_id}_{channel}.nii.gz"
                        std_path = case_output_dir / std_name
                        try:
                            self.logger.info(f"Attempting to rename: {orig_file} -> {std_path}")
                            os.rename(orig_file, std_path)
                            output_files["ADC"] = std_path
                            self.logger.info(f"Successfully renamed ADC file")
                        except Exception as rename_error:
                            self.logger.error(f"Failed to rename ADC file {orig_file} to {std_path}: {rename_error}")
                            return False, {}
                        # --- End ---
                        # Create empty DWI and FLAIR based on ADC
                        dwi_path = case_output_dir / f"ISLES2022_{case_id}_{CHANNEL_MAPPING['DWI']}.nii.gz"
                        flair_path = case_output_dir / f"ISLES2022_{case_id}_{CHANNEL_MAPPING['FLAIR']}.nii.gz"
                        if self.create_empty_nifti(output_files["ADC"], dwi_path):
                            output_files["DWI"] = dwi_path
                        else:
                            return False, {}
                        if self.create_empty_nifti(output_files["ADC"], flair_path):
                            output_files["FLAIR"] = flair_path
                        else:
                            return False, {}
                        return True, output_files
                    else:
                        self.logger.error(f"Failed to convert ADC for case {case_name}")
                        return False, {}
                else:
                    self.logger.error(f"Missing or empty ADC directory for case {case_name}")
                    return False, {}
                    
            elif status.strategy == "empty_flair":
                # Convert ADC and DWI normally, create empty FLAIR
                for modality in ['ADC', 'DWI']:
                    modality_dir = case_dir / modality
                    if modality_dir.exists():
                        success, files = self.convert_dicom_to_nifti(modality_dir, case_output_dir)
                        if success and files:
                            # --- Rename ---
                            orig_file = files[0]
                            channel = CHANNEL_MAPPING[modality]
                            std_name = f"ISLES2022_{case_id}_{channel}.nii.gz"
                            std_path = case_output_dir / std_name
                            try:
                                self.logger.info(f"Attempting to rename: {orig_file} -> {std_path}")
                                os.rename(orig_file, std_path)
                                output_files[modality] = std_path
                                self.logger.info(f"Successfully renamed {modality} file")
                            except Exception as rename_error:
                                self.logger.error(f"Failed to rename {modality} file {orig_file} to {std_path}: {rename_error}")
                                return False, {}
                            # --- End ---
                        else:
                            self.logger.error(f"Failed to convert {modality} for case {case_name}")
                            return False, {}
                    else:
                        self.logger.error(f"Missing {modality} directory for case {case_name}")
                        return False, {}
                
                # Create empty FLAIR based on ADC
                flair_path = case_output_dir / f"ISLES2022_{case_id}_{CHANNEL_MAPPING['FLAIR']}.nii.gz"
                if self.create_empty_nifti(output_files["ADC"], flair_path):
                    output_files["FLAIR"] = flair_path
                    return True, output_files
                else:
                    return False, {}
                    
            elif status.strategy == "empty_dwi":
                # Convert ADC and FLAIR, DWI always uses zero NIfTI, no longer attempt real conversion
                for modality in ['ADC', 'FLAIR']:
                    modality_dir = case_dir / modality
                    if modality_dir.exists():
                        success, files = self.convert_dicom_to_nifti(modality_dir, case_output_dir)
                        if success and files:
                            # --- Rename ---
                            orig_file = files[0]
                            channel = CHANNEL_MAPPING[modality]
                            std_name = f"ISLES2022_{case_id}_{channel}.nii.gz"
                            std_path = case_output_dir / std_name
                            try:
                                self.logger.info(f"Attempting to rename: {orig_file} -> {std_path}")
                                os.rename(orig_file, std_path)
                                output_files[modality] = std_path
                                self.logger.info(f"Successfully renamed {modality} file")
                            except Exception as rename_error:
                                self.logger.error(f"Failed to rename {modality} file {orig_file} to {std_path}: {rename_error}")
                                return False, {}
                            # --- End ---
                        else:
                            self.logger.error(f"Failed to convert {modality} for case {case_name}")
                            return False, {}
                    else:
                        self.logger.error(f"Missing {modality} directory for case {case_name}")
                        return False, {}
                # DWI always uses zero NIfTI (using ADC as reference)
                dwi_path = case_output_dir / f"ISLES2022_{case_id}_{CHANNEL_MAPPING['DWI']}.nii.gz"
                if self.create_empty_nifti(output_files["ADC"], dwi_path):
                    output_files["DWI"] = dwi_path
                    return True, output_files
                else:
                    return False, {}
                    
            elif status.strategy == "empty_adc":
                # Convert DWI and FLAIR normally, create empty ADC
                for modality in ['DWI', 'FLAIR']:
                    modality_dir = case_dir / modality
                    if modality_dir.exists():
                        success, files = self.convert_dicom_to_nifti(modality_dir, case_output_dir)
                        if success and files:
                            # --- Rename ---
                            orig_file = files[0]
                            channel = CHANNEL_MAPPING[modality]
                            std_name = f"ISLES2022_{case_id}_{channel}.nii.gz"
                            std_path = case_output_dir / std_name
                            try:
                                self.logger.info(f"Attempting to rename: {orig_file} -> {std_path}")
                                os.rename(orig_file, std_path)
                                output_files[modality] = std_path
                                self.logger.info(f"Successfully renamed {modality} file")
                            except Exception as rename_error:
                                self.logger.error(f"Failed to rename {modality} file {orig_file} to {std_path}: {rename_error}")
                                return False, {}
                            # --- End ---
                        else:
                            self.logger.error(f"Failed to convert {modality} for case {case_name}")
                            return False, {}
                    else:
                        self.logger.error(f"Missing {modality} directory for case {case_name}")
                        return False, {}
                
                # Create empty ADC based on DWI
                adc_path = case_output_dir / f"ISLES2022_{case_id}_{CHANNEL_MAPPING['ADC']}.nii.gz"
                if self.create_empty_nifti(output_files["DWI"], adc_path):
                    output_files["ADC"] = adc_path
                    return True, output_files
                else:
                    return False, {}
                    
            elif status.strategy == "flair_only":
                # Only FLAIR exists, convert FLAIR normally and create empty ADC and DWI
                flair_dir = case_dir / "FLAIR"
                if flair_dir.exists() and status.flair_count > 0:
                    success, files = self.convert_dicom_to_nifti(flair_dir, case_output_dir)
                    if success and files:
                        # Rename FLAIR
                        orig_file = files[0]
                        channel = CHANNEL_MAPPING["FLAIR"]
                        std_name = f"ISLES2022_{case_id}_{channel}.nii.gz"
                        std_path = case_output_dir / std_name
                        try:
                            self.logger.info(f"Attempting to rename: {orig_file} -> {std_path}")
                            os.rename(orig_file, std_path)
                            output_files["FLAIR"] = std_path
                            self.logger.info(f"Successfully renamed FLAIR file")
                        except Exception as rename_error:
                            self.logger.error(f"Failed to rename FLAIR file {orig_file} to {std_path}: {rename_error}")
                            return False, {}
                        
                        # Create empty ADC and DWI based on FLAIR
                        adc_path = case_output_dir / f"ISLES2022_{case_id}_{CHANNEL_MAPPING['ADC']}.nii.gz"
                        dwi_path = case_output_dir / f"ISLES2022_{case_id}_{CHANNEL_MAPPING['DWI']}.nii.gz"
                        
                        if self.create_empty_nifti(output_files["FLAIR"], adc_path):
                            output_files["ADC"] = adc_path
                        else:
                            return False, {}
                            
                        if self.create_empty_nifti(output_files["FLAIR"], dwi_path):
                            output_files["DWI"] = dwi_path
                        else:
                            return False, {}
                            
                        return True, output_files
                    else:
                        self.logger.error(f"Failed to convert FLAIR for case {case_name}")
                        return False, {}
                else:
                    self.logger.error(f"Missing or empty FLAIR directory for case {case_name}")
                    return False, {}
                    
            elif status.strategy == "dwi_only":
                # Only DWI exists, convert DWI normally and create empty ADC and FLAIR
                dwi_dir = case_dir / "DWI"
                if dwi_dir.exists() and status.dwi_count > 0:
                    success, files = self.convert_dicom_to_nifti(dwi_dir, case_output_dir)
                    if success and files:
                        # Rename DWI
                        orig_file = files[0]
                        channel = CHANNEL_MAPPING["DWI"]
                        std_name = f"ISLES2022_{case_id}_{channel}.nii.gz"
                        std_path = case_output_dir / std_name
                        try:
                            self.logger.info(f"Attempting to rename: {orig_file} -> {std_path}")
                            os.rename(orig_file, std_path)
                            output_files["DWI"] = std_path
                            self.logger.info(f"Successfully renamed DWI file")
                        except Exception as rename_error:
                            self.logger.error(f"Failed to rename DWI file {orig_file} to {std_path}: {rename_error}")
                            return False, {}
                        
                        # Create empty ADC and FLAIR based on DWI
                        adc_path = case_output_dir / f"ISLES2022_{case_id}_{CHANNEL_MAPPING['ADC']}.nii.gz"
                        flair_path = case_output_dir / f"ISLES2022_{case_id}_{CHANNEL_MAPPING['FLAIR']}.nii.gz"
                        
                        if self.create_empty_nifti(output_files["DWI"], adc_path):
                            output_files["ADC"] = adc_path
                        else:
                            return False, {}
                            
                        if self.create_empty_nifti(output_files["DWI"], flair_path):
                            output_files["FLAIR"] = flair_path
                        else:
                            return False, {}
                            
                        return True, output_files
                    else:
                        self.logger.error(f"Failed to convert DWI for case {case_name}")
                        return False, {}
                else:
                    self.logger.error(f"Missing or empty DWI directory for case {case_name}")
                    return False, {}
            
            # Default case - should not reach here
            self.logger.error(f"Unhandled strategy for case {case_name}: {status.strategy}")
            return False, {}
    
    def process_patient_directory(self, patient_dir: Path) -> Tuple[bool, List[Dict[str, Path]]]:
        """
        Process a patient directory containing one or more case directories
        
        Args:
            patient_dir: Directory containing case subdirectories
            
        Returns:
            Tuple of (success, list of dictionaries mapping modality to output file path)
        """
        patient_name = patient_dir.name
        self.logger.info(f"Processing patient: {patient_name}")
        
        # Check if patient has any examination directories
        exam_dirs = [d for d in patient_dir.iterdir() if d.is_dir()]
        if not exam_dirs:
            self.logger.warning(f"No examination directories found for patient: {patient_name}")
            return False, []
        
        # Process each examination
        results = []
        all_exams_successful = True
        
        for exam_dir in exam_dirs:
            success, output_files = self.process_case(exam_dir)
            if success:
                results.append(output_files)
            else:
                all_exams_successful = False
                break
        
        if not all_exams_successful:
            # Move patient to skipped directory
            skipped_patient_dir = self.skipped_dir / patient_name
            if skipped_patient_dir.exists():
                shutil.rmtree(skipped_patient_dir)
            
            # Create a copy in the skipped directory
            shutil.copytree(patient_dir, skipped_patient_dir)
            self.logger.info(f"Copied inconsistent patient to skipped: {patient_name}")
            
            return False, []
        
        return True, results
    
    def process_directory(self, parallel: bool = False, max_workers: int = 4) -> ProcessingStats:
        """
        Process all patient directories in the input directory
        
        Args:
            parallel: Whether to process patients in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            Processing statistics
        """
        start_time = time.time()
        
        if not self.input_dir.exists():
            error_msg = f"Input directory does not exist: {self.input_dir}"
            self.logger.error(error_msg)
            raise PipelineError("dicom_to_nifti", "N/A", error_msg, False)
        
        # Find all patient directories
        patient_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        total_patients = len(patient_dirs)
        
        self.logger.info(f"Found {total_patients} patients to process")
        
        # Initialize statistics
        stats = ProcessingStats(
            total_cases=total_patients,
            successful_cases=0,
            failed_cases=0,
            processing_time=0.0,
            error_details=[]
        )
        
        if total_patients == 0:
            self.logger.warning(f"No patient directories found in {self.input_dir}")
            stats.processing_time = time.time() - start_time
            return stats
        
        # Process patients
        if parallel and total_patients > 1:
            self.logger.info(f"Processing patients in parallel with {max_workers} workers")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_patient = {
                    executor.submit(self.process_patient_directory, patient_dir): patient_dir 
                    for patient_dir in patient_dirs
                }
                
                # Process results as they complete
                for future in as_completed(future_to_patient):
                    patient_dir = future_to_patient[future]
                    try:
                        success, _ = future.result()
                        if success:
                            stats.successful_cases += 1
                        else:
                            stats.failed_cases += 1
                            stats.error_details.append(f"Failed to process patient: {patient_dir.name}")
                    except Exception as e:
                        stats.failed_cases += 1
                        error_msg = f"Exception processing {patient_dir.name}: {e}"
                        stats.error_details.append(error_msg)
                        self.logger.error(error_msg)
        else:
            self.logger.info("Processing patients sequentially")
            for i, patient_dir in enumerate(patient_dirs, 1):
                self.logger.info(f"Processing patient {i}/{total_patients}: {patient_dir.name}")
                
                try:
                    success, _ = self.process_patient_directory(patient_dir)
                    if success:
                        stats.successful_cases += 1
                    else:
                        stats.failed_cases += 1
                        stats.error_details.append(f"Failed to process patient: {patient_dir.name}")
                except Exception as e:
                    stats.failed_cases += 1
                    error_msg = f"Exception processing {patient_dir.name}: {e}"
                    stats.error_details.append(error_msg)
                    self.logger.error(error_msg)
        
        # Calculate processing time
        stats.processing_time = time.time() - start_time
        
        # Log summary
        self.logger.info(f"Conversion summary:")
        self.logger.info(f"  Total patients: {stats.total_cases}")
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