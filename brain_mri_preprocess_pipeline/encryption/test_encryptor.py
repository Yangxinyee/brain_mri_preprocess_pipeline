#!/usr/bin/env python3
"""
Test script for the CaseNumberEncryptor class.
"""

import os
import logging
from pathlib import Path
from brain_mri_preprocess_pipeline.encryption.case_number_encryptor import CaseNumberEncryptor

def main():
    """Test the CaseNumberEncryptor class"""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create a temporary mapping file
    mapping_file = Path("test_mapping.json")
    
    # Create an encryptor
    encryptor = CaseNumberEncryptor(
        encryption_key="test_key",
        mapping_file=mapping_file,
        logger=logger
    )
    
    # Test case numbers
    case_numbers = [
        "10005023486",
        "20000205207",
        "30000156511",
        "70000015680"
    ]
    
    # Encrypt case numbers
    encrypted_numbers = []
    for case_number in case_numbers:
        encrypted = encryptor.encrypt_case_number(case_number)
        encrypted_numbers.append(encrypted)
        logger.info(f"Original: {case_number}, Encrypted: {encrypted}")
    
    # Verify encryption is consistent
    logger.info("\nVerifying encryption consistency...")
    for case_number in case_numbers:
        encrypted = encryptor.encrypt_case_number(case_number)
        logger.info(f"Original: {case_number}, Encrypted: {encrypted}")
    
    # Decrypt case numbers
    logger.info("\nDecrypting case numbers...")
    for encrypted in encrypted_numbers:
        decrypted = encryptor.decrypt_case_number(encrypted)
        logger.info(f"Encrypted: {encrypted}, Decrypted: {decrypted}")
    
    # Test loading and saving mapping
    logger.info("\nTesting mapping persistence...")
    
    # Create a new encryptor with the same mapping file
    new_encryptor = CaseNumberEncryptor(
        encryption_key="test_key",
        mapping_file=mapping_file,
        logger=logger
    )
    
    # Verify decryption still works
    for encrypted in encrypted_numbers:
        decrypted = new_encryptor.decrypt_case_number(encrypted)
        logger.info(f"Encrypted: {encrypted}, Decrypted: {decrypted}")
    
    # Clean up
    if mapping_file.exists():
        os.remove(mapping_file)
        logger.info(f"Removed test mapping file: {mapping_file}")

if __name__ == "__main__":
    main()