#!/usr/bin/env python3
"""
Data models for the medical image processing pipeline.
Defines common data structures used throughout the pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any

@dataclass
class ProcessingStats:
    """Class for tracking processing statistics"""
    total_cases: int = 0
    successful_cases: int = 0
    failed_cases: int = 0
    processing_time: float = 0.0
    error_details: List[str] = field(default_factory=list)

@dataclass
class ModalityStatus:
    """Class for tracking modality consistency"""
    adc_count: int = 0
    dwi_count: int = 0
    flair_count: int = 0
    is_consistent: bool = False
    missing_modalities: List[str] = field(default_factory=list)
    strategy: str = "normal"  # "normal", "empty_flair", "empty_dwi", etc.
    
    def determine_strategy(self) -> str:
        """Determine the conversion strategy based on modality counts"""
        # All modalities have the same count and are present
        if self.adc_count > 0 and self.adc_count == self.dwi_count == self.flair_count:
            self.is_consistent = True
            self.strategy = "normal"
            return self.strategy
        
        # All modalities have different counts
        if len(set([self.adc_count, self.dwi_count, self.flair_count])) == 3:
            self.is_consistent = False
            self.strategy = "adc_only"
            return self.strategy
        
        # ADC and DWI match, but FLAIR differs
        if self.adc_count > 0 and self.adc_count == self.dwi_count and self.adc_count != self.flair_count:
            self.is_consistent = False
            self.strategy = "empty_flair"
            return self.strategy
        
        # ADC and FLAIR match, but DWI differs
        if self.adc_count > 0 and self.adc_count == self.flair_count and self.adc_count != self.dwi_count:
            self.is_consistent = False
            self.strategy = "empty_dwi"
            return self.strategy
        
        # DWI and FLAIR match, but ADC differs
        if self.dwi_count > 0 and self.dwi_count == self.flair_count and self.dwi_count != self.adc_count:
            self.is_consistent = False
            self.strategy = "empty_adc"
            return self.strategy
        
        # Only FLAIR exists
        if self.flair_count > 0 and self.adc_count == 0 and self.dwi_count == 0:
            self.is_consistent = False
            self.strategy = "flair_only"
            return self.strategy
            
        # Only DWI exists
        if self.dwi_count > 0 and self.adc_count == 0 and self.flair_count == 0:
            self.is_consistent = False
            self.strategy = "dwi_only"
            return self.strategy
        
        # Default to ADC only if we get here
        self.is_consistent = False
        self.strategy = "adc_only"
        return self.strategy

@dataclass
class CaseInfo:
    """Class for storing case information"""
    original_path: Path
    case_number: str
    encrypted_case_number: str = ""
    patient_name: str = ""
    study_description: str = ""
    modalities_present: List[str] = field(default_factory=list)
    
    def extract_info_from_path(self) -> None:
        """Extract patient name and study description from path"""
        # Path format: .../patient_id patient_name/case_number study_description/...
        path_parts = self.original_path.parts
        
        # Find the part containing the patient name
        for i, part in enumerate(path_parts):
            if self.case_number in part and i > 0:
                # The previous part should contain patient ID and name
                patient_part = path_parts[i-1]
                # Extract patient name (everything after the first space)
                if " " in patient_part:
                    self.patient_name = patient_part.split(" ", 1)[1]
                
                # The current part should contain case number and study description
                study_part = part
                # Extract study description (everything after the first space)
                if " " in study_part:
                    self.study_description = study_part.split(" ", 1)[1]
                
                break

class PipelineError(Exception):
    """Custom exception for pipeline errors"""
    def __init__(self, step: str, case_id: str, message: str, recoverable: bool = True):
        self.step = step
        self.case_id = case_id
        self.recoverable = recoverable
        super().__init__(f"[{step}] Error processing case {case_id}: {message}")

class ErrorHandler:
    """Handles errors in the pipeline"""
    
    @staticmethod
    def handle_error(error: PipelineError, logger: Optional[Any] = None) -> bool:
        """
        Handle a pipeline error
        
        Args:
            error: The error to handle
            logger: Logger to use for logging the error
            
        Returns:
            True if processing should continue, False otherwise
        """
        if logger:
            logger.error(str(error))
        
        return error.recoverable
    
    @staticmethod
    def should_continue_processing(error: PipelineError) -> bool:
        """Determine if processing should continue after an error"""
        return error.recoverable
    
    @staticmethod
    def log_error_with_context(error: PipelineError, logger: Any) -> None:
        """Log an error with context information"""
        logger.error(f"Error in step '{error.step}' for case '{error.case_id}'")
        logger.error(f"Error message: {str(error)}")
        logger.error(f"Recoverable: {error.recoverable}")

# Constants
CHANNEL_MAPPING = {
    "DWI": "0000",
    "ADC": "0001",
    "FLAIR": "0002"
}