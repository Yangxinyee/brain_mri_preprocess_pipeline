#!/usr/bin/env python3
"""
Unit tests for the DICOM to NIfTI converter module.
"""

import unittest
import os
import shutil
import tempfile
import nibabel as nib
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from brain_mri_preprocess_pipeline.conversion.dicom_to_nifti import DicomToNiftiConverter
from brain_mri_preprocess_pipeline.utils.models import ProcessingStats, ModalityStatus
from brain_mri_preprocess_pipeline.utils.logging_utils import LogManager

class TestDicomToNiftiConverter(unittest.TestCase):
    """Tests for the DicomToNiftiConverter class"""
    
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
        
        # Create converter
        self.converter = DicomToNiftiConverter(self.input_dir, self.output_dir, self.log_manager)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test initialization"""
        self.assertEqual(self.converter.input_dir, self.input_dir)
        self.assertEqual(self.converter.output_dir, self.output_dir)
        self.log_manager.setup_step_logger.assert_called_once_with("dicom_to_nifti")
        self.assertTrue(self.converter.skipped_dir.exists())
    
    def test_check_modality_consistency_all_consistent(self):
        """Test modality consistency check with all modalities consistent"""
        # Create test case directory with modality subdirectories
        case_dir = self.input_dir / "test_case"
        case_dir.mkdir()
        
        for modality in ["ADC", "DWI", "FLAIR"]:
            modality_dir = case_dir / modality
            modality_dir.mkdir()
            # Create 10 DICOM files in each modality directory
            for i in range(10):
                dcm_file = modality_dir / f"file{i}.dcm"
                with open(dcm_file, 'w') as f:
                    f.write(f"Test DICOM file {i}")
        
        # Test consistency check
        status = self.converter.check_modality_consistency(case_dir)
        
        # Verify results
        self.assertEqual(status.adc_count, 10)
        self.assertEqual(status.dwi_count, 10)
        self.assertEqual(status.flair_count, 10)
        self.assertTrue(status.is_consistent)
        self.assertEqual(status.missing_modalities, [])
        self.assertEqual(status.strategy, "normal")
    
    def test_check_modality_consistency_missing_modality(self):
        """Test modality consistency check with a missing modality"""
        # Create test case directory with modality subdirectories
        case_dir = self.input_dir / "test_case"
        case_dir.mkdir()
        
        # Create only ADC and DWI directories
        for modality in ["ADC", "DWI"]:
            modality_dir = case_dir / modality
            modality_dir.mkdir()
            # Create 10 DICOM files in each modality directory
            for i in range(10):
                dcm_file = modality_dir / f"file{i}.dcm"
                with open(dcm_file, 'w') as f:
                    f.write(f"Test DICOM file {i}")
        
        # Test consistency check
        status = self.converter.check_modality_consistency(case_dir)
        
        # Verify results
        self.assertEqual(status.adc_count, 10)
        self.assertEqual(status.dwi_count, 10)
        self.assertEqual(status.flair_count, 0)
        self.assertFalse(status.is_consistent)
        self.assertEqual(status.missing_modalities, ["FLAIR"])
        self.assertEqual(status.strategy, "empty_flair")
        
    def test_check_modality_consistency_flair_only(self):
        """Test modality consistency check with only FLAIR sequence"""
        # Create test case directory with only FLAIR subdirectory
        case_dir = self.input_dir / "test_case_flair_only"
        case_dir.mkdir()
        
        # Create only FLAIR directory
        flair_dir = case_dir / "FLAIR"
        flair_dir.mkdir()
        # Create 10 DICOM files in FLAIR directory
        for i in range(10):
            dcm_file = flair_dir / f"file{i}.dcm"
            with open(dcm_file, 'w') as f:
                f.write(f"Test DICOM file {i}")
        
        # Test consistency check
        status = self.converter.check_modality_consistency(case_dir)
        
        # Verify results
        self.assertEqual(status.adc_count, 0)
        self.assertEqual(status.dwi_count, 0)
        self.assertEqual(status.flair_count, 10)
        self.assertFalse(status.is_consistent)
        self.assertEqual(set(status.missing_modalities), {"ADC", "DWI"})
        self.assertEqual(status.strategy, "flair_only")
        
    def test_check_modality_consistency_dwi_only(self):
        """Test modality consistency check with only DWI sequence"""
        # Create test case directory with only DWI subdirectory
        case_dir = self.input_dir / "test_case_dwi_only"
        case_dir.mkdir()
        
        # Create only DWI directory
        dwi_dir = case_dir / "DWI"
        dwi_dir.mkdir()
        # Create 10 DICOM files in DWI directory
        for i in range(10):
            dcm_file = dwi_dir / f"file{i}.dcm"
            with open(dcm_file, 'w') as f:
                f.write(f"Test DICOM file {i}")
        
        # Test consistency check
        status = self.converter.check_modality_consistency(case_dir)
        
        # Verify results
        self.assertEqual(status.adc_count, 0)
        self.assertEqual(status.dwi_count, 10)
        self.assertEqual(status.flair_count, 0)
        self.assertFalse(status.is_consistent)
        self.assertEqual(set(status.missing_modalities), {"ADC", "FLAIR"})
        self.assertEqual(status.strategy, "dwi_only")
    
    def test_check_modality_consistency_inconsistent_counts(self):
        """Test modality consistency check with inconsistent file counts"""
        # Create test case directory with modality subdirectories
        case_dir = self.input_dir / "test_case"
        case_dir.mkdir()
        
        # Create modality directories with different file counts
        modality_counts = {"ADC": 10, "DWI": 5, "FLAIR": 8}
        for modality, count in modality_counts.items():
            modality_dir = case_dir / modality
            modality_dir.mkdir()
            # Create DICOM files in each modality directory
            for i in range(count):
                dcm_file = modality_dir / f"file{i}.dcm"
                with open(dcm_file, 'w') as f:
                    f.write(f"Test DICOM file {i}")
        
        # Test consistency check
        status = self.converter.check_modality_consistency(case_dir)
        
        # Verify results
        self.assertEqual(status.adc_count, 10)
        self.assertEqual(status.dwi_count, 5)
        self.assertEqual(status.flair_count, 8)
        self.assertFalse(status.is_consistent)
        self.assertEqual(status.missing_modalities, [])
        self.assertEqual(status.strategy, "adc_only")
    
    @patch('subprocess.run')
    def test_convert_dicom_to_nifti_success(self, mock_run):
        """Test successful DICOM to NIfTI conversion"""
        # Set up mock subprocess.run to return success
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Conversion successful"
        mock_process.stderr = ""
        mock_run.return_value = mock_process
        
        # Create test DICOM directory
        dicom_dir = self.input_dir / "test_dicom"
        dicom_dir.mkdir()
        
        # Create a test output directory
        output_dir = self.output_dir / "test_output"
        
        # Create a mock NIfTI file that would be created by dcm2niix
        output_dir.mkdir()
        nifti_file = output_dir / "test_nifti.nii.gz"
        with open(nifti_file, 'w') as f:
            f.write("Test NIfTI file")
        
        # Test conversion
        success, files = self.converter.convert_dicom_to_nifti(dicom_dir, output_dir)
        
        # Verify results
        self.assertTrue(success)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0], nifti_file)
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_convert_dicom_to_nifti_failure(self, mock_run):
        """Test failed DICOM to NIfTI conversion"""
        # Set up mock subprocess.run to return failure
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stdout = ""
        mock_process.stderr = "Conversion failed"
        mock_run.return_value = mock_process
        
        # Create test DICOM directory
        dicom_dir = self.input_dir / "test_dicom"
        dicom_dir.mkdir()
        
        # Create a test output directory
        output_dir = self.output_dir / "test_output"
        output_dir.mkdir()
        
        # Test conversion
        success, files = self.converter.convert_dicom_to_nifti(dicom_dir, output_dir)
        
        # Verify results
        self.assertFalse(success)
        self.assertEqual(len(files), 0)
        mock_run.assert_called_once()
    
    @patch('nibabel.load')
    @patch('nibabel.save')
    def test_create_empty_nifti(self, mock_save, mock_load):
        """Test creating an empty NIfTI file"""
        # Create mock NIfTI image
        mock_img = MagicMock()
        mock_img.shape = (10, 10, 10)
        mock_img.affine = np.eye(4)
        mock_img.header = MagicMock()
        mock_load.return_value = mock_img
        
        # Create test reference file
        ref_file = self.output_dir / "reference.nii.gz"
        with open(ref_file, 'w') as f:
            f.write("Test reference file")
        
        # Test creating empty NIfTI
        output_file = self.output_dir / "empty.nii.gz"
        result = self.converter.create_empty_nifti(ref_file, output_file)
        
        # Verify results
        self.assertTrue(result)
        mock_load.assert_called_once_with(ref_file)
        mock_save.assert_called_once()
    
    @patch.object(DicomToNiftiConverter, 'convert_dicom_to_nifti')
    @patch.object(DicomToNiftiConverter, 'create_empty_nifti')
    def test_process_case_consistent(self, mock_create_empty, mock_convert):
        """Test processing a case with consistent modalities"""
        # Set up mocks
        mock_convert.return_value = (True, [Path("test.nii.gz")])
        
        # Create test case directory with modality subdirectories
        case_dir = self.input_dir / "test_case"
        case_dir.mkdir()
        
        for modality in ["ADC", "DWI", "FLAIR"]:
            modality_dir = case_dir / modality
            modality_dir.mkdir()
            # Create 10 DICOM files in each modality directory
            for i in range(10):
                dcm_file = modality_dir / f"file{i}.dcm"
                with open(dcm_file, 'w') as f:
                    f.write(f"Test DICOM file {i}")
        
        # Test processing case
        success, files = self.converter.process_case(case_dir)
        
        # Verify results
        self.assertTrue(success)
        self.assertEqual(len(files), 3)
        self.assertEqual(mock_convert.call_count, 3)
        mock_create_empty.assert_not_called()
    
    @patch.object(DicomToNiftiConverter, 'convert_dicom_to_nifti')
    @patch.object(DicomToNiftiConverter, 'create_empty_nifti')
    def test_process_case_inconsistent(self, mock_create_empty, mock_convert):
        """Test processing a case with inconsistent modalities"""
        # Set up mocks
        mock_convert.return_value = (True, [Path("test.nii.gz")])
        mock_create_empty.return_value = True
        
        # Create test case directory with modality subdirectories
        case_dir = self.input_dir / "test_case"
        case_dir.mkdir()
        
        # Create only ADC directory
        modality_dir = case_dir / "ADC"
        modality_dir.mkdir()
        # Create 10 DICOM files
        for i in range(10):
            dcm_file = modality_dir / f"file{i}.dcm"
            with open(dcm_file, 'w') as f:
                f.write(f"Test DICOM file {i}")
        
        # Test processing case
        success, files = self.converter.process_case(case_dir)
        
        # Verify results
        self.assertTrue(success)
        self.assertEqual(len(files), 3)
        self.assertEqual(mock_convert.call_count, 1)
        self.assertEqual(mock_create_empty.call_count, 2)

if __name__ == "__main__":
    unittest.main()