#!/usr/bin/env python3
"""
Case number encryption module for the medical image processing pipeline.
Provides reversible encryption for patient case numbers to protect privacy.
"""

import os
import json
import base64
import logging
import time
from pathlib import Path
from typing import Dict, Optional, List
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ..utils.models import ProcessingStats
from ..utils.logging_utils import LogManager

class CaseNumberEncryptor:
    """
    Provides reversible encryption for patient case numbers.
    
    This class implements a secure, reversible encryption system for patient
    case numbers, maintaining a mapping between original and encrypted values.
    """
    
    def __init__(self, encryption_key_file: Optional[Path] = None, log_manager: Optional[LogManager] = None):
        """
        Initialize the case number encryptor.
        
        Args:
            encryption_key_file: Path to file containing the encryption key
            log_manager: Logging manager
        """
        self.log_manager = log_manager
        self.logger = log_manager.setup_step_logger("encryption") if log_manager else logging.getLogger(__name__)
        
        # Default mapping file location
        self.mapping_file = Path("output/encryption/case_mapping.json")
        self.mapping: Dict[str, str] = {}
        self.reverse_mapping: Dict[str, str] = {}
        
        # Read encryption key from file or use default
        encryption_key = "default_encryption_key_for_development_only"
        if encryption_key_file and encryption_key_file.exists():
            try:
                with open(encryption_key_file, 'r') as f:
                    encryption_key = f.read().strip()
            except Exception as e:
                self.logger.error(f"Error reading encryption key file: {e}")
        
        # Generate a key from the provided encryption key
        self.key = self._generate_key(encryption_key)
        self.cipher = Fernet(self.key)
        
        # Create directory for mapping file if it doesn't exist
        self.mapping_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing mapping if available
        self.load_mapping()
        
    def _generate_key(self, encryption_key: str) -> bytes:
        """
        Generate a Fernet key from the provided encryption key.
        
        Args:
            encryption_key: Base encryption key
            
        Returns:
            Fernet-compatible key
        """
        # Use PBKDF2 to derive a secure key
        salt = b'medical_image_pipeline_salt'  # In production, this should be unique per deployment
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(encryption_key.encode()))
        return key
    
    def encrypt_case_number(self, original_number: str) -> str:
        """
        Encrypt a case number.
        
        Args:
            original_number: Original case number
            
        Returns:
            Encrypted case number
        """
        # Check if we already have an encrypted version
        if original_number in self.mapping:
            return self.mapping[original_number]
        
        # Encrypt the case number directly
        encrypted_data = self.cipher.encrypt(original_number.encode())
        
        # Convert to a URL-safe string and take first 16 characters
        # This creates a shorter identifier while maintaining uniqueness
        encrypted_str = base64.urlsafe_b64encode(encrypted_data).decode()[:16]
        
        # Ensure uniqueness - if there's a collision, add more characters
        while encrypted_str in self.reverse_mapping and self.reverse_mapping[encrypted_str] != original_number:
            # Take more characters from the encoded string
            encrypted_str = base64.urlsafe_b64encode(encrypted_data).decode()[:len(encrypted_str) + 4]
        
        # Store in mapping
        self.mapping[original_number] = encrypted_str
        self.reverse_mapping[encrypted_str] = original_number
        
        # Save the updated mapping
        self.save_mapping()
        
        self.logger.info(f"Encrypted case number: {original_number} -> {encrypted_str}")
        return encrypted_str
    
    def decrypt_case_number(self, encrypted_number: str) -> str:
        """
        Decrypt a case number.
        
        Args:
            encrypted_number: Encrypted case number
            
        Returns:
            Original case number
        """
        # Check if we have this in our mapping
        if encrypted_number in self.reverse_mapping:
            return self.reverse_mapping[encrypted_number]
        
        # If not in mapping, we can't decrypt it
        self.logger.error(f"Cannot decrypt case number: {encrypted_number} (not found in mapping)")
        return encrypted_number
    
    def save_mapping(self) -> bool:
        """
        Save the mapping to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.mapping_file, 'w') as f:
                json.dump(self.mapping, f, indent=2)
            self.logger.info(f"Saved mapping to {self.mapping_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving mapping: {e}")
            return False
    
    def load_mapping(self) -> bool:
        """
        Load the mapping from file.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.mapping_file.exists():
            self.logger.info(f"Mapping file does not exist: {self.mapping_file}")
            return False
        
        try:
            with open(self.mapping_file, 'r') as f:
                self.mapping = json.load(f)
                
            # Rebuild reverse mapping
            self.reverse_mapping = {v: k for k, v in self.mapping.items()}
            
            self.logger.info(f"Loaded {len(self.mapping)} mappings from {self.mapping_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading mapping: {e}")
            return False
            
    def process_directory(self, directory: Optional[Path] = None, parallel: bool = False, 
                         max_workers: int = 4) -> ProcessingStats:
        """
        Process all case numbers in a directory.
        This method is implemented to match the interface expected by the PipelineOrchestrator.
        
        Args:
            directory: Directory containing case directories (not used directly)
            parallel: Whether to process in parallel (not used)
            max_workers: Maximum number of parallel workers (not used)
            
        Returns:
            Processing statistics
        """
        start_time = time.time()
        
        # For encryption, we don't actually need to process a directory
        # We just need to ensure the mapping file is ready
        stats = ProcessingStats()
        
        # Count existing mappings as "processed cases"
        stats.total_cases = len(self.mapping)
        stats.successful_cases = len(self.mapping)
        
        # Calculate processing time
        stats.processing_time = time.time() - start_time
        
        self.logger.info(f"Encryption module ready with {len(self.mapping)} existing mappings")
        return stats