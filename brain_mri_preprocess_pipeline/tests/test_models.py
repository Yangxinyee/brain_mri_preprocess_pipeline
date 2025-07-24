#!/usr/bin/env python3
"""
Unit tests for the models module.
"""

import unittest
from pathlib import Path
from brain_mri_preprocess_pipeline.utils.models import (
    ProcessingStats, ModalityStatus, CaseInfo, PipelineError, ErrorHandler, CHANNEL_MAPPING
)

class TestProcessingStats(unittest.TestCase):
    """Tests for the ProcessingStats class"""
    
    def test_initialization(self):
        """Test initialization with default values"""
        stats = ProcessingStats()
        self.assertEqual(stats.total_cases, 0)
        self.assertEqual(stats.successful_cases, 0)
        self.assertEqual(stats.failed_cases, 0)
        self.assertEqual(stats.processing_time, 0.0)
        self.assertEqual(stats.error_details, [])
    
    def test_initialization_with_values(self):
        """Test initialization with custom values"""
        stats = ProcessingStats(
            total_cases=10,
            successful_cases=8,
            failed_cases=2,
            processing_time=5.5,
            error_details=["Error 1", "Error 2"]
        )
        self.assertEqual(stats.total_cases, 10)
        self.assertEqual(stats.successful_cases, 8)
        self.assertEqual(stats.failed_cases, 2)
        self.assertEqual(stats.processing_time, 5.5)
        self.assertEqual(stats.error_details, ["Error 1", "Error 2"])

class TestModalityStatus(unittest.TestCase):
    """Tests for the ModalityStatus class"""
    
    def test_initialization(self):
        """Test initialization with default values"""
        status = ModalityStatus()
        self.assertEqual(status.adc_count, 0)
        self.assertEqual(status.dwi_count, 0)
        self.assertEqual(status.flair_count, 0)
        self.assertFalse(status.is_consistent)
        self.assertEqual(status.missing_modalities, [])
        self.assertEqual(status.strategy, "normal")
    
    def test_determine_strategy_all_consistent(self):
        """Test strategy determination when all modalities have the same count"""
        status = ModalityStatus(adc_count=10, dwi_count=10, flair_count=10)
        strategy = status.determine_strategy()
        self.assertTrue(status.is_consistent)
        self.assertEqual(strategy, "normal")
        self.assertEqual(status.strategy, "normal")
    
    def test_determine_strategy_all_different(self):
        """Test strategy determination when all modalities have different counts"""
        status = ModalityStatus(adc_count=10, dwi_count=5, flair_count=0)
        strategy = status.determine_strategy()
        self.assertFalse(status.is_consistent)
        self.assertEqual(strategy, "adc_only")
        self.assertEqual(status.strategy, "adc_only")
    
    def test_determine_strategy_flair_differs(self):
        """Test strategy determination when ADC and DWI match but FLAIR differs"""
        status = ModalityStatus(adc_count=10, dwi_count=10, flair_count=5)
        strategy = status.determine_strategy()
        self.assertFalse(status.is_consistent)
        self.assertEqual(strategy, "empty_flair")
        self.assertEqual(status.strategy, "empty_flair")
    
    def test_determine_strategy_dwi_differs(self):
        """Test strategy determination when ADC and FLAIR match but DWI differs"""
        status = ModalityStatus(adc_count=10, dwi_count=5, flair_count=10)
        strategy = status.determine_strategy()
        self.assertFalse(status.is_consistent)
        self.assertEqual(strategy, "empty_dwi")
        self.assertEqual(status.strategy, "empty_dwi")
    
    def test_determine_strategy_adc_differs(self):
        """Test strategy determination when DWI and FLAIR match but ADC differs"""
        status = ModalityStatus(adc_count=5, dwi_count=10, flair_count=10)
        strategy = status.determine_strategy()
        self.assertFalse(status.is_consistent)
        self.assertEqual(strategy, "empty_adc")
        self.assertEqual(status.strategy, "empty_adc")
    
    def test_determine_strategy_missing_modalities(self):
        """Test strategy determination with missing modalities"""
        status = ModalityStatus(adc_count=0, dwi_count=0, flair_count=0)
        status.missing_modalities = ["ADC", "DWI", "FLAIR"]
        strategy = status.determine_strategy()
        self.assertFalse(status.is_consistent)
        self.assertEqual(strategy, "adc_only")
        self.assertEqual(status.strategy, "adc_only")

class TestCaseInfo(unittest.TestCase):
    """Tests for the CaseInfo class"""
    
    def test_initialization(self):
        """Test initialization with required values"""
        case_info = CaseInfo(
            original_path=Path("/path/to/case"),
            case_number="12345"
        )
        self.assertEqual(case_info.original_path, Path("/path/to/case"))
        self.assertEqual(case_info.case_number, "12345")
        self.assertEqual(case_info.encrypted_case_number, "")
        self.assertEqual(case_info.patient_name, "")
        self.assertEqual(case_info.study_description, "")
        self.assertEqual(case_info.modalities_present, [])
    
    def test_extract_info_from_path(self):
        """Test extracting information from path"""
        # Path format: .../patient_id patient_name/case_number study_description/...
        case_info = CaseInfo(
            original_path=Path("/data/10005023486 MONROE DARLENE E/12345 BRAIN MRI/FLAIR"),
            case_number="12345"
        )
        case_info.extract_info_from_path()
        self.assertEqual(case_info.patient_name, "MONROE DARLENE E")
        self.assertEqual(case_info.study_description, "BRAIN MRI")

class TestPipelineError(unittest.TestCase):
    """Tests for the PipelineError class"""
    
    def test_initialization(self):
        """Test initialization with required values"""
        error = PipelineError("conversion", "12345", "Failed to convert")
        self.assertEqual(error.step, "conversion")
        self.assertEqual(error.case_id, "12345")
        self.assertTrue(error.recoverable)
        self.assertEqual(str(error), "[conversion] Error processing case 12345: Failed to convert")
    
    def test_initialization_non_recoverable(self):
        """Test initialization with non-recoverable error"""
        error = PipelineError("conversion", "12345", "Critical failure", False)
        self.assertEqual(error.step, "conversion")
        self.assertEqual(error.case_id, "12345")
        self.assertFalse(error.recoverable)
        self.assertEqual(str(error), "[conversion] Error processing case 12345: Critical failure")

class TestErrorHandler(unittest.TestCase):
    """Tests for the ErrorHandler class"""
    
    def test_should_continue_processing_recoverable(self):
        """Test should_continue_processing with recoverable error"""
        error = PipelineError("conversion", "12345", "Failed to convert")
        self.assertTrue(ErrorHandler.should_continue_processing(error))
    
    def test_should_continue_processing_non_recoverable(self):
        """Test should_continue_processing with non-recoverable error"""
        error = PipelineError("conversion", "12345", "Critical failure", False)
        self.assertFalse(ErrorHandler.should_continue_processing(error))

class TestChannelMapping(unittest.TestCase):
    """Tests for the CHANNEL_MAPPING constant"""
    
    def test_channel_mapping(self):
        """Test channel mapping values"""
        self.assertEqual(CHANNEL_MAPPING["DWI"], "0000")
        self.assertEqual(CHANNEL_MAPPING["ADC"], "0001")
        self.assertEqual(CHANNEL_MAPPING["FLAIR"], "0002")

if __name__ == "__main__":
    unittest.main()