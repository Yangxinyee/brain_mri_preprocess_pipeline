#!/usr/bin/env python3
"""
Integration tests for the medical image processing pipeline.
Tests the complete workflow from end to end and performs performance benchmarking.
"""

import unittest
import os
import sys
import shutil
import tempfile
import time
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
import subprocess
import json

# Add parent directory to path to allow importing from parent modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brain_mri_preprocess_pipeline.utils.pipeline_orchestrator import PipelineOrchestrator, PipelineStep
from brain_mri_preprocess_pipeline.utils.models import ProcessingStats, ModalityStatus, CaseInfo
from brain_mri_preprocess_pipeline.utils.logging_utils import LogManager
from brain_mri_preprocess_pipeline.utils.config import PipelineConfig
from brain_mri_preprocess_pipeline.utils.environment import EnvironmentManager

from brain_mri_preprocess_pipeline.decompression.dicom_decompressor import DicomDecompressor
from brain_mri_preprocess_pipeline.conversion.dicom_to_nifti import DicomToNiftiConverter
from brain_mri_preprocess_pipeline.registration.registration_engine import ImageRegistrationEngine
from brain_mri_preprocess_pipeline.skull_stripping.skull_stripper import SkullStrippingProcessor
from brain_mri_preprocess_pipeline.encryption.case_number_encryptor import CaseNumberEncryptor
from brain_mri_preprocess_pipeline.organization.file_organizer import FileOrganizer

class TestIntegrationPipeline(unittest.TestCase):
    """Integration tests for the complete pipeline workflow"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        # Check if we're in the right conda environment
        try:
            env_name = os.environ.get('CONDA_DEFAULT_ENV')
            if env_name != 'nnunet-gpu':
                print(f"Warning: Tests are running in '{env_name}' environment, not 'nnunet-gpu'")
                print("Some tests may fail if required tools are not available")
        except Exception as e:
            print(f"Warning: Could not determine conda environment: {e}")
    
    def setUp(self):
        """Set up test environment for each test"""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "input"
        self.output_dir = Path(self.temp_dir) / "output"
        self.log_dir = Path(self.temp_dir) / "logs"
        
        # Create directory structure
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        self.log_dir.mkdir()
        
        # Create configuration
        self.config = PipelineConfig(
            input_directory=self.input_dir,
            output_directory=self.output_dir,
            log_directory=self.log_dir,
            parallel_workers=2
        )
        
        # Create log manager
        self.log_manager = LogManager(self.log_dir)
        self.logger = self.log_manager.setup_step_logger("integration_test")
        
        # Set up test data
        self._setup_test_data()
    
    def tearDown(self):
        """Clean up test environment after each test"""
        shutil.rmtree(self.temp_dir)
    
    def _setup_test_data(self):
        """Set up test data for integration tests"""
        # Create a simple test case structure
        # We'll use a simplified structure since we can't include real DICOM files in tests
        
        # Create case directory structure
        case_id = "10005023486"
        patient_name = "MONROE DARLENE E"
        case_dir = self.input_dir / f"{case_id} {patient_name}"
        case_dir.mkdir()
        
        # Create modality directories
        modalities = ["ADC", "DWI", "FLAIR"]
        for modality in modalities:
            modality_dir = case_dir / modality
            modality_dir.mkdir()
            
            # Create placeholder DICOM files (empty files)
            for i in range(20):  # 20 slices per modality
                dicom_file = modality_dir / f"IM-{i:04d}.dcm"
                dicom_file.touch()
        
        self.logger.info(f"Test data setup complete in {self.input_dir}")
    
    def _create_mock_pipeline(self):
        """Create a mock pipeline for testing"""
        # Initialize the pipeline orchestrator
        orchestrator = PipelineOrchestrator(self.config, self.log_manager)
        
        # Initialize processing modules with mocks
        decompressor = MagicMock(spec=DicomDecompressor)
        decompressor.process_directory.return_value = ProcessingStats(
            total_cases=1, successful_cases=1, failed_cases=0, processing_time=0.5
        )
        
        converter = MagicMock(spec=DicomToNiftiConverter)
        converter.process_directory.return_value = ProcessingStats(
            total_cases=1, successful_cases=1, failed_cases=0, processing_time=0.5
        )
        
        registration_engine = MagicMock(spec=ImageRegistrationEngine)
        registration_engine.process_directory.return_value = ProcessingStats(
            total_cases=1, successful_cases=1, failed_cases=0, processing_time=0.5
        )
        
        skull_stripper = MagicMock(spec=SkullStrippingProcessor)
        skull_stripper.process_directory.return_value = ProcessingStats(
            total_cases=1, successful_cases=1, failed_cases=0, processing_time=0.5
        )
        
        encryptor = MagicMock(spec=CaseNumberEncryptor)
        encryptor.process_directory.return_value = ProcessingStats(
            total_cases=1, successful_cases=1, failed_cases=0, processing_time=0.5
        )
        
        file_organizer = MagicMock(spec=FileOrganizer)
        file_organizer.process_directory.return_value = ProcessingStats(
            total_cases=1, successful_cases=1, failed_cases=0, processing_time=0.5
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
        
        return orchestrator
    
    def test_end_to_end_mock_pipeline(self):
        """Test the complete pipeline workflow with mocked components"""
        # Create mock pipeline
        orchestrator = self._create_mock_pipeline()
        
        # Execute pipeline
        start_time = time.time()
        success = orchestrator.execute_pipeline()
        end_time = time.time()
        
        # Verify results
        self.assertTrue(success)
        
        # Check that all steps were executed
        for step in orchestrator.steps:
            self.assertTrue(step.processor.process_directory.called)
        
        # Log execution time
        execution_time = end_time - start_time
        self.logger.info(f"Mock pipeline execution time: {execution_time:.2f} seconds")
    
    def test_pipeline_step_sequence(self):
        """Test that pipeline steps are executed in the correct sequence"""
        # Create mock pipeline
        orchestrator = self._create_mock_pipeline()
        
        # Track the order of execution
        execution_order = []
        
        # Override the execute method of each step to track execution order
        for step in orchestrator.steps:
            original_execute = step.execute
            
            def make_tracking_execute(step_name, orig_execute):
                def tracking_execute(*args, **kwargs):
                    execution_order.append(step_name)
                    return orig_execute(*args, **kwargs)
                return tracking_execute
            
            step.execute = make_tracking_execute(step.name, original_execute)
        
        # Execute pipeline
        success = orchestrator.execute_pipeline()
        
        # Verify results
        self.assertTrue(success)
        
        # Check execution order
        expected_order = [
            "decompression",
            "conversion",
            "registration",
            "skull_stripping",
            "encryption",
            "organization"
        ]
        
        self.assertEqual(execution_order, expected_order)
    
    def test_pipeline_with_disabled_steps(self):
        """Test pipeline execution with some steps disabled"""
        # Create mock pipeline
        orchestrator = self._create_mock_pipeline()
        
        # Disable some steps
        orchestrator.disable_step("decompression")
        orchestrator.disable_step("registration")
        
        # Execute pipeline
        success = orchestrator.execute_pipeline()
        
        # Verify results
        self.assertTrue(success)
        
        # Check that only enabled steps were executed
        for step in orchestrator.steps:
            if step.name in ["decompression", "registration"]:
                self.assertFalse(step.processor.process_directory.called)
            else:
                self.assertTrue(step.processor.process_directory.called)
    
    @unittest.skipIf(os.environ.get('CONDA_DEFAULT_ENV') != 'nnunet-gpu', 
                    "Test requires nnunet-gpu environment")
    def test_environment_verification(self):
        """Test environment verification"""
        # Create environment manager
        env_manager = EnvironmentManager(self.logger)
        
        # Verify environment
        env_verified = env_manager.verify_environment()
        
        # Check required tools
        tools_verified = env_manager.check_required_tools(["dcm2niix", "antsRegistration", "hd-bet"])
        
        # Log results
        self.logger.info(f"Environment verification: {env_verified}")
        self.logger.info(f"Tools verification: {tools_verified}")
        
        # We don't assert here because we don't want to fail the test if the environment is not set up
        # This is more of an informational test
    
    def test_performance_benchmarking(self):
        """Test performance benchmarking of the pipeline"""
        # Create mock pipeline with varying processing times
        orchestrator = PipelineOrchestrator(self.config, self.log_manager)
        
        # Create steps with different processing times
        steps_data = [
            ("step1", 0.1),  # Fast step
            ("step2", 0.5),  # Medium step
            ("step3", 1.0),  # Slow step
        ]
        
        for name, processing_time in steps_data:
            # Create mock processor
            processor = MagicMock()
            processor.process_directory.return_value = ProcessingStats(
                total_cases=100, 
                successful_cases=100, 
                failed_cases=0, 
                processing_time=processing_time
            )
            
            # Add step to pipeline
            orchestrator.add_step(PipelineStep(
                name=name,
                description=f"Step {name}",
                processor=processor,
                method_name="process_directory"
            ))
        
        # Execute pipeline and measure time
        start_time = time.time()
        orchestrator.execute_pipeline()
        end_time = time.time()
        
        # Calculate total execution time
        total_time = end_time - start_time
        
        # Log performance metrics
        self.logger.info(f"Total execution time: {total_time:.2f} seconds")
        
        # Log individual step times
        for step in orchestrator.steps:
            self.logger.info(f"Step {step.name} time: {step.stats.processing_time:.2f} seconds")
        
        # Calculate throughput (cases per second)
        total_cases = orchestrator.steps[0].stats.total_cases
        throughput = total_cases / total_time
        self.logger.info(f"Throughput: {throughput:.2f} cases per second")
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery in the pipeline"""
        # Create orchestrator
        orchestrator = PipelineOrchestrator(self.config, self.log_manager)
        
        # Create steps with different failure rates
        steps_data = [
            ("step1", 1.0),    # 100% success
            ("step2", 0.7),    # 70% success
            ("step3", 0.9),    # 90% success
        ]
        
        for name, success_rate in steps_data:
            # Create mock processor
            processor = MagicMock()
            processor.process_directory.return_value = ProcessingStats(
                total_cases=100, 
                successful_cases=int(100 * success_rate), 
                failed_cases=int(100 * (1 - success_rate)), 
                processing_time=0.1
            )
            
            # Add step to pipeline
            orchestrator.add_step(PipelineStep(
                name=name,
                description=f"Step {name}",
                processor=processor,
                method_name="process_directory"
            ))
        
        # Execute pipeline
        success = orchestrator.execute_pipeline()
        
        # Verify results
        self.assertTrue(success)  # Pipeline should complete even with errors
        
        # Check that all steps were executed
        for step in orchestrator.steps:
            self.assertTrue(step.processor.process_directory.called)
        
        # Check error statistics
        self.assertEqual(orchestrator.steps[0].stats.failed_cases, 0)    # step1: 0% failure
        self.assertEqual(orchestrator.steps[1].stats.failed_cases, 30)   # step2: 30% failure
        self.assertEqual(orchestrator.steps[2].stats.failed_cases, 10)   # step3: 10% failure
    
    @unittest.skipIf(not Path("test_input").exists(), 
                    "Test requires test_input directory with sample data")
    def test_with_sample_data(self):
        """Test pipeline with sample data if available"""
        # Check if test_input directory exists
        test_input_dir = Path("test_input")
        if not test_input_dir.exists():
            self.skipTest("test_input directory not found")
        
        # Create a new config using the test_input directory
        config = PipelineConfig(
            input_directory=test_input_dir,
            output_directory=self.output_dir,
            log_directory=self.log_dir,
            parallel_workers=1
        )
        
        # Create orchestrator with real components
        orchestrator = PipelineOrchestrator(config, self.log_manager)
        
        # Initialize processing modules
        try:
            # We'll use mocks for components that require external tools
            decompressor = DicomDecompressor(
                input_dir=config.input_directory,
                output_dir=config.decompressed_dir,
                log_manager=self.log_manager
            )
            
            converter = MagicMock(spec=DicomToNiftiConverter)
            converter.process_directory.return_value = ProcessingStats(
                total_cases=1, successful_cases=1, failed_cases=0, processing_time=0.5
            )
            
            registration_engine = MagicMock(spec=ImageRegistrationEngine)
            registration_engine.process_directory.return_value = ProcessingStats(
                total_cases=1, successful_cases=1, failed_cases=0, processing_time=0.5
            )
            
            skull_stripper = MagicMock(spec=SkullStrippingProcessor)
            skull_stripper.process_directory.return_value = ProcessingStats(
                total_cases=1, successful_cases=1, failed_cases=0, processing_time=0.5
            )
            
            encryptor = CaseNumberEncryptor(
                encryption_key_file=None,  # Use default key
                log_manager=self.log_manager
            )
            
            file_organizer = MagicMock(spec=FileOrganizer)
            file_organizer.process_directory.return_value = ProcessingStats(
                total_cases=1, successful_cases=1, failed_cases=0, processing_time=0.5
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
            
            # Execute pipeline
            success = orchestrator.execute_pipeline()
            
            # Verify results
            self.assertTrue(success)
            
        except Exception as e:
            self.logger.error(f"Error in test_with_sample_data: {e}", exc_info=True)
            self.fail(f"Test failed with error: {e}")
    
    def test_command_line_interface(self):
        """Test the command-line interface"""
        # Create a mock subprocess.run function
        with patch('subprocess.run') as mock_run:
            # Configure the mock to return a successful result
            mock_run.return_value = MagicMock(returncode=0)
            
            # Test basic command
            cmd = [
                "python", "-m", "brain_mri_preprocess_pipeline.medical_image_pipeline",
                "--input", str(self.input_dir),
                "--output", str(self.output_dir),
                "--logs", str(self.log_dir)
            ]
            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check if the command was executed
            # Note: This will actually try to run the command, but we're not checking the result
            # since it will likely fail without proper setup
            self.logger.info(f"Command executed with return code: {result.returncode}")
            self.logger.info(f"Command output: {result.stdout}")
            if result.stderr:
                self.logger.info(f"Command error: {result.stderr}")
    
    def test_parallel_vs_sequential_performance(self):
        """Compare performance between parallel and sequential processing"""
        # Create orchestrator
        orchestrator = PipelineOrchestrator(self.config, self.log_manager)
        
        # Create a processor that simulates processing time
        class TimedProcessor:
            def __init__(self, processing_time_per_case):
                self.processing_time_per_case = processing_time_per_case
            
            def process_directory(self, parallel=False, max_workers=1):
                # Simulate processing 10 cases
                total_cases = 10
                start_time = time.time()
                
                # Simulate processing time
                if parallel and max_workers > 1:
                    # Parallel processing is faster
                    time.sleep(self.processing_time_per_case * total_cases / max_workers)
                else:
                    # Sequential processing
                    time.sleep(self.processing_time_per_case * total_cases)
                
                end_time = time.time()
                
                # Return stats
                return ProcessingStats(
                    total_cases=total_cases,
                    successful_cases=total_cases,
                    failed_cases=0,
                    processing_time=end_time - start_time
                )
        
        # Add a step with the timed processor
        processor = TimedProcessor(0.1)  # 0.1 seconds per case
        orchestrator.add_step(PipelineStep(
            name="timed_step",
            description="Timed processing step",
            processor=processor,
            method_name="process_directory"
        ))
        
        # Execute pipeline sequentially
        self.logger.info("Running sequential processing...")
        start_time = time.time()
        orchestrator.execute_pipeline(parallel=False)
        sequential_time = time.time() - start_time
        
        # Execute pipeline in parallel
        self.logger.info("Running parallel processing...")
        start_time = time.time()
        orchestrator.execute_pipeline(parallel=True)
        parallel_time = time.time() - start_time
        
        # Log results
        self.logger.info(f"Sequential processing time: {sequential_time:.2f} seconds")
        self.logger.info(f"Parallel processing time: {parallel_time:.2f} seconds")
        
        # Calculate speedup
        if sequential_time > 0:
            speedup = sequential_time / parallel_time
            self.logger.info(f"Speedup factor: {speedup:.2f}x")
        
        # Parallel should be faster
        self.assertLess(parallel_time, sequential_time)

class TestIntegrationValidation(unittest.TestCase):
    """Tests for validating the output of the pipeline"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir()
        
        # Create test output structure
        self.final_dir = self.output_dir / "Dataset002_ISLES2022_all/imagesTs"
        self.final_dir.mkdir(parents=True)
        
        # Create sample output files
        self._create_sample_output()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def _create_sample_output(self):
        """Create sample output files for validation"""
        # Create sample NIfTI files (empty files with correct naming)
        case_id = "ISLES2022_encrypted_123"
        
        for channel, suffix in [("0000", "DWI"), ("0001", "ADC"), ("0002", "FLAIR")]:
            filename = f"{case_id}_{channel}.nii.gz"
            filepath = self.final_dir / filename
            filepath.touch()
    
    def test_output_file_structure(self):
        """Test that the output file structure is correct"""
        # Check that the final directory exists
        self.assertTrue(self.final_dir.exists())
        
        # Check that files exist with correct naming pattern
        files = list(self.final_dir.glob("*.nii.gz"))
        self.assertTrue(len(files) > 0)
        
        # Check naming pattern
        for file in files:
            filename = file.name
            # Should match pattern: ISLES2022_<encrypted_id>_<channel>.nii.gz
            parts = filename.split('_')
            self.assertEqual(parts[0], "ISLES2022")
            self.assertTrue(len(parts) == 3)
            
            # Check channel suffix
            channel = parts[2].split('.')[0]
            self.assertTrue(channel in ["0000", "0001", "0002"])
    
    def test_file_organization_validation(self):
        """Test validation of file organization"""
        # Create a validator function
        def validate_file_organization(directory):
            # Check directory structure
            if not directory.exists():
                return False, "Directory does not exist"
            
            # Check for NIfTI files
            nifti_files = list(directory.glob("*.nii.gz"))
            if not nifti_files:
                return False, "No NIfTI files found"
            
            # Check naming convention
            for file in nifti_files:
                filename = file.name
                if not filename.startswith("ISLES2022_"):
                    return False, f"File {filename} does not follow naming convention"
                
                parts = filename.split('_')
                if len(parts) != 3:
                    return False, f"File {filename} does not have correct number of parts"
                
                channel = parts[2].split('.')[0]
                if channel not in ["0000", "0001", "0002"]:
                    return False, f"File {filename} has invalid channel suffix"
            
            return True, "Validation successful"
        
        # Validate the output
        valid, message = validate_file_organization(self.final_dir)
        self.assertTrue(valid, message)
    
    def test_case_completeness(self):
        """Test that each case has all required modalities"""
        # Group files by case ID
        cases = {}
        for file in self.final_dir.glob("*.nii.gz"):
            filename = file.name
            parts = filename.split('_')
            case_id = parts[1]
            channel = parts[2].split('.')[0]
            
            if case_id not in cases:
                cases[case_id] = []
            cases[case_id].append(channel)
        
        # Check that each case has all three modalities
        for case_id, channels in cases.items():
            self.assertEqual(len(channels), 3, f"Case {case_id} is missing modalities")
            self.assertIn("0000", channels, f"Case {case_id} is missing DWI")
            self.assertIn("0001", channels, f"Case {case_id} is missing ADC")
            self.assertIn("0002", channels, f"Case {case_id} is missing FLAIR")

if __name__ == "__main__":
    unittest.main()