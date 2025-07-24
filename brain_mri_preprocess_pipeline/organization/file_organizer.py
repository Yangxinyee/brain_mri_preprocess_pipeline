#!/usr/bin/env python3
"""
File organization module for the medical image processing pipeline.
Organizes processed files into standardized dataset structure with consistent naming.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

from brain_mri_preprocess_pipeline.utils.models import CaseInfo, CHANNEL_MAPPING, PipelineError, ProcessingStats
from brain_mri_preprocess_pipeline.encryption.case_number_encryptor import CaseNumberEncryptor

class FileOrganizer:
    """
    Organizes processed files into standardized dataset structure.
    
    This class implements the file organization system that renames and
    structures processed files according to the standardized naming convention.
    """
    
    def __init__(self, input_dir: Path, output_dir: Path, encryptor: CaseNumberEncryptor, 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the file organizer.
        
        Args:
            input_dir: Directory containing processed files
            output_dir: Directory where organized files will be stored
            encryptor: Case number encryptor for generating encrypted case IDs
            logger: Logger instance for logging operations
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.encryptor = encryptor
        self.logger = logger or logging.getLogger(__name__)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset structure constants
        self.dataset_name = "ISLES2022"
    
    def organize_case(self, case_dir: Path) -> bool:
        """
        Organize files for a specific case.
        
        Args:
            case_dir: Directory containing processed files for a case
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract case number from directory path
            case_number = self._extract_case_number(case_dir)
            if not case_number:
                self.logger.error(f"Could not extract case number from path: {case_dir}")
                return False
            
            # Create case info
            case_info = CaseInfo(original_path=case_dir, case_number=case_number)
            
            # Encrypt case number
            encrypted_case_id = self.encryptor.encrypt_case_number(case_number)
            case_info.encrypted_case_number = encrypted_case_id
            
            # Extract additional info from path
            case_info.extract_info_from_path()
            
            # Find modality files
            modality_files = self._find_modality_files(case_dir)
            if not modality_files:
                self.logger.error(f"No modality files found for case: {case_number}")
                return False
            
            # Organize files
            success = self._organize_modality_files(case_info, modality_files)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error organizing case {case_dir}: {str(e)}")
            return False
    
    def _extract_case_number(self, case_dir: Path) -> str:
        """
        Extract case number from directory path.
        
        Args:
            case_dir: Directory path
            
        Returns:
            Case number as string
        """
        # Try to extract from the path parts
        for part in case_dir.parts:
            # Case numbers are typically numeric IDs at the beginning of directory names
            if " " in part:
                # Split by space and check if first part is numeric
                first_part = part.split(" ")[0]
                if first_part.isdigit():
                    return first_part
        
        # If we couldn't find a case number, use the directory name as fallback
        return case_dir.name
    
    def _find_modality_files(self, case_dir: Path) -> Dict[str, Path]:
        """
        Find all modality files for a case (supports skull_stripped suffix and 0002_registered)
        Args:
            case_dir: Directory containing files for a case
        Returns:
            Dict of modality name to file path
        """
        import re
        modality_files = {}
        for file_path in case_dir.rglob("*.nii.gz"):
            # Support skull_stripped and registered suffixes
            m = re.match(r"ISLES2022_(\d+)_(0000|0001|0002(?:_registered)?)_skull_stripped\.nii\.gz", file_path.name)
            if m:
                modality = m.group(2)
                # Normalize registered suffix
                if modality == "0002_registered":
                    modality = "0002"
                modality_files[modality] = file_path
        return modality_files
    
    def _organize_modality_files(self, case_info: CaseInfo, modality_files: Dict[str, Path]) -> bool:
        """
        Organize modality files according to standardized naming convention.
        
        Args:
            case_info: Case information
            modality_files: Dictionary mapping modality names to file paths
            
        Returns:
            True if successful, False otherwise
        """
        success = True
        
        for modality, file_path in modality_files.items():
            try:
                # Generate final filename
                final_filename = self.generate_final_filename(case_info.encrypted_case_number, modality)
                
                # Create destination path
                dest_path = self.output_dir / final_filename
                
                # Copy file to destination
                shutil.copy2(file_path, dest_path)
                
                self.logger.info(f"Organized file: {file_path} -> {dest_path}")
                
                # Add modality to present modalities
                if modality not in case_info.modalities_present:
                    case_info.modalities_present.append(modality)
                
            except Exception as e:
                self.logger.error(f"Error organizing file {file_path}: {str(e)}")
                success = False
        
        return success
    
    def generate_final_filename(self, encrypted_case_id: str, modality: str) -> str:
        """
        Generate final filename according to standardized naming convention.
        
        Args:
            encrypted_case_id: Encrypted case ID
            modality: Modality name (DWI, ADC, FLAIR, 0000, 0001, 0002)
        Returns:
            Final filename
        """
        # If modality is already 0000/0001/0002, use directly, otherwise look up CHANNEL_MAPPING
        if modality in {"0000", "0001", "0002"}:
            channel_suffix = modality
        else:
            from brain_mri_preprocess_pipeline.utils.models import CHANNEL_MAPPING
            channel_suffix = CHANNEL_MAPPING.get(modality, "0000")
        filename = f"{self.dataset_name}_{encrypted_case_id}_{channel_suffix}.nii.gz"
        return filename
    
    def process_directory(self, directory: Path = None, parallel: bool = False, 
                         max_workers: int = 4) -> ProcessingStats:
        """
        Process all cases in a directory.
        
        Args:
            directory: Directory containing case directories (defaults to input_dir)
            parallel: Whether to process in parallel (not used in this implementation)
            max_workers: Maximum number of parallel workers (not used in this implementation)
            
        Returns:
            Processing statistics
        """
        import re
        start_time = time.time()
        directory = directory or self.input_dir
        stats = ProcessingStats()
        # Recursively find all nii.gz files
        all_files = list(directory.rglob("*.nii.gz"))
        case_to_files = {}
        for file_path in all_files:
            # Extract case number
            m = re.match(r"ISLES2022_(\d+)_", file_path.name)
            if m:
                case_number = m.group(1)
                case_to_files.setdefault(case_number, []).append(file_path)
        for case_number, files in case_to_files.items():
            stats.total_cases += 1
            # Create temporary case directory
            from tempfile import TemporaryDirectory
            with TemporaryDirectory() as tmpdir:
                tmp_case_dir = Path(tmpdir) / case_number
                tmp_case_dir.mkdir(parents=True, exist_ok=True)
                for f in files:
                    shutil.copy(f, tmp_case_dir / f.name)
                if self.organize_case(tmp_case_dir):
                    stats.successful_cases += 1
                else:
                    stats.failed_cases += 1
                    stats.error_details.append(f"Failed to organize case: {case_number}")
        stats.processing_time = time.time() - start_time
        self.logger.info(f"Processed {stats.total_cases} cases, {stats.successful_cases} successful, {stats.failed_cases} failed")
        self.logger.info(f"Processing time: {stats.processing_time:.2f} seconds")
        return stats
    
    def preserve_original_structure(self, reference_dir: Path, output_dir: Path) -> bool:
        """
        Preserve the original directory structure as reference.
        
        Args:
            reference_dir: Original directory structure
            output_dir: Directory where reference structure will be stored
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a simple text file with directory structure
            structure_file = output_dir / "original_structure.txt"
            
            with open(structure_file, 'w') as f:
                f.write(f"Original directory structure from: {reference_dir}\n")
                f.write("=" * 80 + "\n\n")
                
                # Write directory structure
                for path in sorted(reference_dir.glob("**/*")):
                    rel_path = path.relative_to(reference_dir)
                    indent = "  " * len(rel_path.parts[:-1])
                    f.write(f"{indent}{path.name}\n")
            
            self.logger.info(f"Preserved original structure in: {structure_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error preserving original structure: {str(e)}")
            return False