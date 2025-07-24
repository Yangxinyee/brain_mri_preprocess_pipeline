#!/usr/bin/env python3
"""
Environment management utilities for the medical image processing pipeline.
Handles conda environment verification and tool availability checks.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import List, Optional

# Default conda environment name
DEFAULT_CONDA_ENV = "nnunet-gpu"

class EnvironmentManager:
    """Manages conda environment activation and tool availability verification"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the environment manager"""
        self.logger = logger or logging.getLogger(__name__)
    
    def verify_environment(self, required_env: str = DEFAULT_CONDA_ENV) -> bool:
        """Verify that the correct conda environment is activated"""
        try:
            # Check if conda is available
            result = subprocess.run(
                ["conda", "info", "--envs"], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode != 0:
                self.logger.error("Conda is not available in the system path")
                return False
            
            # Check if the required environment is activated
            env_name = os.environ.get("CONDA_DEFAULT_ENV")
            if env_name != required_env:
                self.logger.error(f"Required conda environment '{required_env}' is not activated. "
                                f"Current environment: '{env_name}'")
                return False
            
            self.logger.info(f"Verified conda environment: {env_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying conda environment: {e}")
            return False
    
    def check_tool_availability(self, tool_name: str) -> bool:
        """Check if a specific tool is available in the current environment"""
        try:
            result = subprocess.run(
                ["which", tool_name], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                tool_path = result.stdout.strip()
                self.logger.info(f"Tool '{tool_name}' found at: {tool_path}")
                return True
            else:
                self.logger.error(f"Tool '{tool_name}' not found in the current environment")
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking tool availability: {e}")
            return False
    
    def check_required_tools(self, tools: List[str]) -> bool:
        """Check if all required tools are available"""
        all_available = True
        
        for tool in tools:
            if not self.check_tool_availability(tool):
                all_available = False
        
        return all_available
    
    def get_tool_path(self, tool_name: str) -> Optional[str]:
        """Get the path to a specific tool"""
        try:
            result = subprocess.run(
                ["which", tool_name], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return None
                
        except Exception:
            return None