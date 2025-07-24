#!/usr/bin/env python3
"""
Unit tests for the DICOM decompressor module.
"""

import unittest
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from brain_mri_preprocess_pipeline.decompression.dicom_decompressor import DicomDecompressor
from brain_mri_preprocess_pipeline.utils.models import ProcessingStats
from brain_mri_preprocess_pipeline.utils.logging_utils import LogManager

class TestDicomDecompressor(unittest.TestCase):
    """Tests for the DicomDecompressor class"""
    
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
        
        # Create decompressor
        self.decompressor = DicomDecompressor(self.input_dir, self.output_dir, self.log_manager)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test initialization"""
        self.assertEqual(self.decompressor.input_dir, self.input_dir)
        self.assertEqual(self.decompressor.output_dir, self.output_dir)
        self.log_manager.setup_step_logger.assert_called_once_with("dicom_decompression")
    
    @patch('pydicom.dcmread')
    def test_decompress_file_success(self, mock_dcmread):
        """Test successful decompression of a file"""
        # Create a mock DICOM dataset
        mock_ds = MagicMock()
        mock_ds.file_meta.TransferSyntaxUID.is_compressed = True
        mock_dcmread.return_value = mock_ds
        
        # Create a test DICOM file
        test_file = self.input_dir / "test.dcm"
        with open(test_file, 'w') as f:
            f.write("Test DICOM file")
        
        # Test decompression
        result = self.decompressor.decompress_file(test_file)
        
        # Verify results
        self.assertTrue(result)
        mock_dcmread.assert_called_once_with(test_file, force=True)
        mock_ds.decompress.assert_called_once()
        mock_ds.save_as.assert_called_once()
    
    @patch('pydicom.dcmread')
    def test_decompress_file_not_compressed(self, mock_dcmread):
        """Test handling of uncompressed file"""
        # Create a mock DICOM dataset
        mock_ds = MagicMock()
        mock_ds.file_meta.TransferSyntaxUID.is_compressed = False
        mock_dcmread.return_value = mock_ds
        
        # Create a test DICOM file
        test_file = self.input_dir / "test.dcm"
        with open(test_file, 'w') as f:
            f.write("Test DICOM file")
        
        # Test decompression
        result = self.decompressor.decompress_file(test_file)
        
        # Verify results
        self.assertTrue(result)
        mock_dcmread.assert_called_once_with(test_file, force=True)
        mock_ds.decompress.assert_not_called()
        mock_ds.save_as.assert_called_once()
    
    @patch('pydicom.dcmread')
    def test_decompress_file_error(self, mock_dcmread):
        """Test error handling during decompression"""
        # Make dcmread raise an exception
        mock_dcmread.side_effect = Exception("Test error")
        
        # Create a test DICOM file
        test_file = self.input_dir / "test.dcm"
        with open(test_file, 'w') as f:
            f.write("Test DICOM file")
        
        # Test decompression
        result = self.decompressor.decompress_file(test_file)
        
        # Verify results
        self.assertFalse(result)
        mock_dcmread.assert_called_once_with(test_file, force=True)
    
    def test_process_directory_empty(self):
        """Test processing an empty directory"""
        # Test processing
        stats = self.decompressor.process_directory()
        
        # Verify results
        self.assertEqual(stats.total_cases, 0)
        self.assertEqual(stats.successful_cases, 0)
        self.assertEqual(stats.failed_cases, 0)
        self.assertGreater(stats.processing_time, 0)
    
    @patch.object(DicomDecompressor, 'decompress_file')
    def test_process_directory_with_files(self, mock_decompress_file):
        """Test processing a directory with files"""
        # Set up mock to return True (success)
        mock_decompress_file.return_value = True
        
        # Create test DICOM files
        for i in range(3):
            test_file = self.input_dir / f"test{i}.dcm"
            with open(test_file, 'w') as f:
                f.write(f"Test DICOM file {i}")
        
        # Test processing
        stats = self.decompressor.process_directory()
        
        # Verify results
        self.assertEqual(stats.total_cases, 3)
        self.assertEqual(stats.successful_cases, 3)
        self.assertEqual(stats.failed_cases, 0)
        self.assertGreater(stats.processing_time, 0)
        self.assertEqual(mock_decompress_file.call_count, 3)
    
    @patch.object(DicomDecompressor, 'decompress_file')
    def test_process_directory_with_failures(self, mock_decompress_file):
        """Test processing a directory with some failures"""
        # Set up mock to alternate between success and failure
        mock_decompress_file.side_effect = [True, False, True]
        
        # Create test DICOM files
        for i in range(3):
            test_file = self.input_dir / f"test{i}.dcm"
            with open(test_file, 'w') as f:
                f.write(f"Test DICOM file {i}")
        
        # Test processing
        stats = self.decompressor.process_directory()
        
        # Verify results
        self.assertEqual(stats.total_cases, 3)
        self.assertEqual(stats.successful_cases, 2)
        self.assertEqual(stats.failed_cases, 1)
        self.assertGreater(stats.processing_time, 0)
        self.assertEqual(mock_decompress_file.call_count, 3)
        self.assertEqual(len(stats.error_details), 1)
    
    def test_process_directory_nonexistent(self):
        """Test processing a nonexistent directory"""
        # Set up nonexistent directory
        nonexistent_dir = Path(self.temp_dir) / "nonexistent"
        decompressor = DicomDecompressor(nonexistent_dir, self.output_dir, self.log_manager)
        
        # Test processing
        with self.assertRaises(Exception):
            decompressor.process_directory()

if __name__ == "__main__":
    unittest.main()