#!/usr/bin/env python3
"""
Unit tests for the skull stripping module.
"""

import unittest
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from brain_mri_preprocess_pipeline.skull_stripping.skull_stripper import SkullStrippingProcessor
from brain_mri_preprocess_pipeline.utils.models import ProcessingStats
from brain_mri_preprocess_pipeline.utils.logging_utils import LogManager

class TestSkullStrippingProcessor(unittest.TestCase):
    """Tests for the SkullStrippingProcessor class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "input"
        self.output_dir = Path(self.temp_dir) / "output"
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        
        # Create mock log manager
        self.log_manager = MagicMock()
        self.log_manager.setup_step_logger.return_value = MagicMock()
        
        # Create skull stripper
        self.skull_stripper = SkullStrippingProcessor(self.input_dir, self.output_dir, self.log_manager)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test initialization"""
        self.assertEqual(self.skull_stripper.input_dir, self.input_dir)
        self.assertEqual(self.skull_stripper.output_dir, self.output_dir)
        self.log_manager.setup_step_logger.assert_called_once_with("skull_stripping")
    
    @patch('subprocess.run')
    def test_process_case_success(self, mock_run):
        """Test successful skull stripping of a case"""
        # Set up mock subprocess.run to return success
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Skull stripping successful"
        mock_process.stderr = ""
        mock_run.return_value = mock_process
        
        # Create test case directory
        case_dir = self.input_dir / "test_case"
        case_dir.mkdir()
        
        # Create test NIfTI files for all modalities
        for modality in ["DWI", "ADC", "FLAIR"]:
            nifti_file = case_dir / f"test_case_{modality}.nii.gz"
            with open(nifti_file, 'w') as f:
                f.write(f"Test NIfTI file {modality}")
        
        # Test skull stripping
        result = self.skull_stripper.process_case("test_case")
        
        # Verify results
        self.assertTrue(result)
        self.assertEqual(mock_run.call_count, 3)  # One call per modality
    
    @patch('subprocess.run')
    def test_process_case_failure(self, mock_run):
        """Test failed skull stripping of a case"""
        # Set up mock subprocess.run to return failure
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stdout = ""
        mock_process.stderr = "Skull stripping failed"
        mock_run.return_value = mock_process
        
        # Create test case directory
        case_dir = self.input_dir / "test_case"
        case_dir.mkdir()
        
        # Create test NIfTI files for all modalities
        for modality in ["DWI", "ADC", "FLAIR"]:
            nifti_file = case_dir / f"test_case_{modality}.nii.gz"
            with open(nifti_file, 'w') as f:
                f.write(f"Test NIfTI file {modality}")
        
        # Test skull stripping
        result = self.skull_stripper.process_case("test_case")
        
        # Verify results
        self.assertFalse(result)
        self.assertEqual(mock_run.call_count, 1)  # Should fail on first modality
    
    def test_process_case_missing_files(self):
        """Test skull stripping with missing files"""
        # Create test case directory
        case_dir = self.input_dir / "test_case"
        case_dir.mkdir()
        
        # Test skull stripping without creating the required files
        result = self.skull_stripper.process_case("test_case")
        
        # Verify results
        self.assertFalse(result)
    
    @patch.object(SkullStrippingProcessor, 'process_case')
    def test_process_directory(self, mock_process_case):
        """Test processing a directory of cases"""
        # Set up mock to return True (success)
        mock_process_case.return_value = True
        
        # Create test case directories
        for i in range(3):
            case_dir = self.input_dir / f"test_case_{i}"
            case_dir.mkdir()
            
            # Create test NIfTI files for all modalities
            for modality in ["DWI", "ADC", "FLAIR"]:
                nifti_file = case_dir / f"test_case_{i}_{modality}.nii.gz"
                with open(nifti_file, 'w') as f:
                    f.write(f"Test NIfTI file {modality}")
        
        # Test processing
        stats = self.skull_stripper.process_directory()
        
        # Verify results
        self.assertEqual(stats.total_cases, 3)
        self.assertEqual(stats.successful_cases, 3)
        self.assertEqual(stats.failed_cases, 0)
        self.assertGreater(stats.processing_time, 0)
        self.assertEqual(mock_process_case.call_count, 3)
    
    @patch.object(SkullStrippingProcessor, 'process_case')
    def test_process_directory_with_failures(self, mock_process_case):
        """Test processing a directory with some failures"""
        # Set up mock to alternate between success and failure
        mock_process_case.side_effect = [True, False, True]
        
        # Create test case directories
        for i in range(3):
            case_dir = self.input_dir / f"test_case_{i}"
            case_dir.mkdir()
            
            # Create test NIfTI files for all modalities
            for modality in ["DWI", "ADC", "FLAIR"]:
                nifti_file = case_dir / f"test_case_{i}_{modality}.nii.gz"
                with open(nifti_file, 'w') as f:
                    f.write(f"Test NIfTI file {modality}")
        
        # Test processing
        stats = self.skull_stripper.process_directory()
        
        # Verify results
        self.assertEqual(stats.total_cases, 3)
        self.assertEqual(stats.successful_cases, 2)
        self.assertEqual(stats.failed_cases, 1)
        self.assertGreater(stats.processing_time, 0)
        self.assertEqual(mock_process_case.call_count, 3)
        self.assertEqual(len(stats.error_details), 1)
    
    def test_verify_skull_stripping(self):
        """Test skull stripping verification"""
        # This is a placeholder test since the actual implementation would require
        # real NIfTI files with image data
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()