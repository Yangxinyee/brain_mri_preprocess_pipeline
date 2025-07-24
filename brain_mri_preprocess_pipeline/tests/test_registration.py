#!/usr/bin/env python3
"""
Test cases for image registration functionality
Tests the handling of zero images and edge cases in registration
"""

import unittest
import tempfile
import shutil
import numpy as np
import nibabel as nib
from pathlib import Path
from unittest.mock import patch, MagicMock

from brain_mri_preprocess_pipeline.registration.registration_engine import ImageRegistrationEngine
from brain_mri_preprocess_pipeline.utils.logging_utils import LogManager


class TestImageRegistrationEngine(unittest.TestCase):
    """Test cases for ImageRegistrationEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directories
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input"
        self.output_dir = self.temp_dir / "output" 
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log manager
        log_manager = LogManager(self.temp_dir / "logs")
        
        # Create registration engine
        self.engine = ImageRegistrationEngine(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            log_manager=log_manager
        )
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def create_test_nifti(self, file_path: Path, data: np.ndarray = None, is_zero: bool = False):
        """Create a test NIfTI file"""
        if data is None:
            if is_zero:
                data = np.zeros((10, 10, 10), dtype=np.float32)
            else:
                data = np.random.rand(10, 10, 10).astype(np.float32)
        
        # Create a simple affine matrix
        affine = np.eye(4)
        
        # Create NIfTI image
        img = nib.Nifti1Image(data, affine)
        
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the image
        nib.save(img, str(file_path))
        
    def test_scenario_1_dwi_normal_flair_zero(self):
        """Test scenario 1: DWI normal, FLAIR zero - skip registration"""
        case_id = "12345"
        case_dir = self.input_dir / f"case_{case_id}"
        
        # Create test files
        dwi_file = case_dir / f"ISLES2022_{case_id}_0000.nii.gz"
        flair_file = case_dir / f"ISLES2022_{case_id}_0002.nii.gz"
        adc_file = case_dir / f"ISLES2022_{case_id}_0001.nii.gz"
        
        self.create_test_nifti(dwi_file, is_zero=False)  # Normal DWI
        self.create_test_nifti(flair_file, is_zero=True)  # Zero FLAIR
        self.create_test_nifti(adc_file, is_zero=False)  # Normal ADC
        
        # Test processing
        result = self.engine.process_case(case_id)
        
        # Verify results
        self.assertTrue(result)
        
        # Check that registered file was created by copying original
        registered_file = case_dir / f"ISLES2022_{case_id}_0002_registered.nii.gz"
        self.assertTrue(registered_file.exists())
        
        # Check that original FLAIR was deleted
        self.assertFalse(flair_file.exists())
        
        # Verify the registered file contains zero data
        registered_img = nib.load(str(registered_file))
        self.assertTrue(np.all(registered_img.get_fdata() == 0))
        
    def test_scenario_3_dwi_zero_flair_zero_adc_normal(self):
        """Test scenario 3: DWI zero, FLAIR zero, ADC normal - skip registration"""
        case_id = "12346"
        case_dir = self.input_dir / f"case_{case_id}"
        
        # Create test files
        dwi_file = case_dir / f"ISLES2022_{case_id}_0000.nii.gz"
        flair_file = case_dir / f"ISLES2022_{case_id}_0002.nii.gz"
        adc_file = case_dir / f"ISLES2022_{case_id}_0001.nii.gz"
        
        self.create_test_nifti(dwi_file, is_zero=True)   # Zero DWI
        self.create_test_nifti(flair_file, is_zero=True) # Zero FLAIR
        self.create_test_nifti(adc_file, is_zero=False)  # Normal ADC
        
        # Test processing
        result = self.engine.process_case(case_id)
        
        # Verify results
        self.assertTrue(result)
        
        # Check that registered file was created
        registered_file = case_dir / f"ISLES2022_{case_id}_0002_registered.nii.gz"
        self.assertTrue(registered_file.exists())
        
        # Check that original FLAIR was deleted
        self.assertFalse(flair_file.exists())
        
    def test_scenario_4_dwi_zero_flair_normal_adc_zero(self):
        """Test scenario 4: DWI zero, FLAIR normal, ADC zero - skip registration"""
        case_id = "12347"
        case_dir = self.input_dir / f"case_{case_id}"
        
        # Create test files
        dwi_file = case_dir / f"ISLES2022_{case_id}_0000.nii.gz"
        flair_file = case_dir / f"ISLES2022_{case_id}_0002.nii.gz"
        adc_file = case_dir / f"ISLES2022_{case_id}_0001.nii.gz"
        
        self.create_test_nifti(dwi_file, is_zero=True)    # Zero DWI
        self.create_test_nifti(flair_file, is_zero=False) # Normal FLAIR
        self.create_test_nifti(adc_file, is_zero=True)    # Zero ADC
        
        # Test processing
        result = self.engine.process_case(case_id)
        
        # Verify results
        self.assertTrue(result)
        
        # Check that registered file was created
        registered_file = case_dir / f"ISLES2022_{case_id}_0002_registered.nii.gz"
        self.assertTrue(registered_file.exists())
        
        # Check that original FLAIR was deleted
        self.assertFalse(flair_file.exists())
        
        # Verify the registered file contains non-zero data (copied from original FLAIR)
        registered_img = nib.load(str(registered_file))
        self.assertFalse(np.all(registered_img.get_fdata() == 0))
        
    @patch.object(ImageRegistrationEngine, 'register_flair_to_dwi')
    def test_scenario_2_dwi_zero_flair_normal_adc_normal(self, mock_register):
        """Test scenario 2: DWI zero, FLAIR normal, ADC normal - use ADC for registration"""
        case_id = "12348"
        case_dir = self.input_dir / f"case_{case_id}"
        
        # Create test files
        dwi_file = case_dir / f"ISLES2022_{case_id}_0000.nii.gz"
        flair_file = case_dir / f"ISLES2022_{case_id}_0002.nii.gz"
        adc_file = case_dir / f"ISLES2022_{case_id}_0001.nii.gz"
        
        self.create_test_nifti(dwi_file, is_zero=True)    # Zero DWI
        self.create_test_nifti(flair_file, is_zero=False) # Normal FLAIR
        self.create_test_nifti(adc_file, is_zero=False)   # Normal ADC
        
        # Mock successful registration
        mock_register.return_value = True
        
        # Mock verify_registration_quality to return True
        with patch.object(self.engine, 'verify_registration_quality', return_value=True):
            # Create mock output file
            output_file = self.output_dir / f"ISLES2022_{case_id}_0002_registered.nii.gz"
            self.create_test_nifti(output_file, is_zero=False)
            
            # Test processing
            result = self.engine.process_case(case_id)
            
        # Verify results
        self.assertTrue(result)
        
        # Verify that register_flair_to_dwi was called with ADC as template
        mock_register.assert_called_once()
        args = mock_register.call_args[0]
        self.assertEqual(str(args[0]), str(adc_file))  # First arg should be ADC file
        
    def test_all_modalities_zero_case_skipped(self):
        """Test that case is skipped when all modalities are zero"""
        case_id = "12349"
        case_dir = self.input_dir / f"case_{case_id}"
        
        # Create test files - all zero
        dwi_file = case_dir / f"ISLES2022_{case_id}_0000.nii.gz"
        flair_file = case_dir / f"ISLES2022_{case_id}_0002.nii.gz"
        adc_file = case_dir / f"ISLES2022_{case_id}_0001.nii.gz"
        
        self.create_test_nifti(dwi_file, is_zero=True)   # Zero DWI
        self.create_test_nifti(flair_file, is_zero=True) # Zero FLAIR
        self.create_test_nifti(adc_file, is_zero=True)   # Zero ADC
        
        # Test processing
        result = self.engine.process_case(case_id)
        
        # Verify case was skipped
        self.assertFalse(result)
        
        # Check that case was moved to skipped_cases
        skipped_dir = self.input_dir.parent.parent / "skipped_cases"
        self.assertTrue(skipped_dir.exists())
        
    @patch.object(ImageRegistrationEngine, 'register_flair_to_dwi')  
    def test_normal_case_uses_dwi_template(self, mock_register):
        """Test normal case uses DWI as registration template"""
        case_id = "12350"
        case_dir = self.input_dir / f"case_{case_id}"
        
        # Create test files - all normal
        dwi_file = case_dir / f"ISLES2022_{case_id}_0000.nii.gz"
        flair_file = case_dir / f"ISLES2022_{case_id}_0002.nii.gz"
        adc_file = case_dir / f"ISLES2022_{case_id}_0001.nii.gz"
        
        self.create_test_nifti(dwi_file, is_zero=False)   # Normal DWI
        self.create_test_nifti(flair_file, is_zero=False) # Normal FLAIR
        self.create_test_nifti(adc_file, is_zero=False)   # Normal ADC
        
        # Mock successful registration
        mock_register.return_value = True
        
        # Mock verify_registration_quality to return True
        with patch.object(self.engine, 'verify_registration_quality', return_value=True):
            # Create mock output file
            output_file = self.output_dir / f"ISLES2022_{case_id}_0002_registered.nii.gz"
            self.create_test_nifti(output_file, is_zero=False)
            
            # Test processing
            result = self.engine.process_case(case_id)
            
        # Verify results
        self.assertTrue(result)
        
        # Verify that register_flair_to_dwi was called with DWI as template
        mock_register.assert_called_once()
        args = mock_register.call_args[0]
        self.assertEqual(str(args[0]), str(dwi_file))  # First arg should be DWI file


if __name__ == '__main__':
    unittest.main()